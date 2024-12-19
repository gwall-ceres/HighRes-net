import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage import transform, img_as_float
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import SimilarityTransform
from skimage.registration import phase_cross_correlation
import registration_metrics as rm



def apply_shift_to_template(shift_x, shift_y, template_image, template_mask):
         # Apply shift to the template image
        shifted_image = ndi_shift(
            template_image,
            shift=(shift_y, shift_x),
            mode='constant',
            cval=0,
            order=3  # Cubic interpolation for images
        )
        shifted_image.flags.writeable = False
        # logging.debug("Applied shift to template image using scipy.ndimage.shift.")
        
        tform = transform.EuclideanTransform(
            rotation=0,
            translation = (-shift_x, -shift_y)
        )
        #shifted_image = tform.inverse(shifted_image)
        shifted_mask = transform.warp(template_mask.astype(float), tform, mode='constant', cval=0, order=1)
        
        # Apply shift to the template mask
        '''
        shifted_mask = ndi_shift(
            self.template_mask_array.astype(float),
            shift=(total_shift_y, total_shift_x),
            mode='constant',
            order=1,
            cval=0
        )
        '''
        #shifted_mask = shifted_mask > 0.5  # Re-binarize the mask
        shifted_mask.flags.writeable = False

        return shifted_image, shifted_mask

def compute_grid_mncc(norm_ref, ref_mask, template_image, template_mask, bounds_y, bounds_x, points_per_dim):
    """
    Compute MNCC scores for a grid of points within given bounds.
    
    Parameters:
        norm_ref (np.ndarray): Normalized reference image
        ref_mask (np.ndarray): Reference mask as float [0,1]
        template_image (np.ndarray): Template image to be aligned
        template_mask (np.ndarray): Template mask as bool
        bounds_y (tuple): (min_y, max_y) bounds for vertical shifts
        bounds_x (tuple): (min_x, max_x) bounds for horizontal shifts
        points_per_dim (int): Number of points to evaluate per dimension
        
    Returns:
        tuple: (best_shift_y, best_shift_x, best_score)
    """
    min_y, max_y = bounds_y
    min_x, max_x = bounds_x
    
    # Generate grid points
    y_points = np.linspace(min_y, max_y, points_per_dim)
    x_points = np.linspace(min_x, max_x, points_per_dim)
    
    best_score = float('-inf')
    best_shift = (0.0, 0.0)
    
    # Evaluate MNCC at each grid point
    for dy in y_points:
        for dx in x_points:
            # Shift template image and mask
            shifted_template, shifted_mask = apply_shift_to_template(dx, dy, template_image, template_mask)
            shifted_mask = (shifted_mask > 0.5).astype(float)
            
            # Compute combined mask
            combined_mask = ref_mask * shifted_mask
            weight_sum = np.sum(combined_mask)
            
            if weight_sum > 0:
                # Compute normalized template statistics
                mu_template = np.sum(shifted_template * combined_mask) / weight_sum
                sigma_template = np.sqrt(np.sum(combined_mask * 
                                              (shifted_template - mu_template) ** 2) / 
                                      weight_sum)
                
                if sigma_template > 0:
                    # Compute correlation using pre-computed normalized reference
                    norm_template = (shifted_template - mu_template) / sigma_template
                    mncc = np.sum(combined_mask * norm_ref * norm_template) / weight_sum
                    
                    if mncc > best_score:
                        best_score = mncc
                        best_shift = (dy, dx)
    
    return best_shift[0], best_shift[1], best_score


def recursive_mncc_search(norm_ref, ref_mask, template_image, template_mask, 
                         points_per_dim, scale_factor, max_recursions, current_recursion=0,
                         prev_best_dy=0.0, prev_best_dx=0.0):
    """
    Recursive function to search for the best alignment using MNCC.
    
    Parameters:
        norm_ref (np.ndarray): Normalized reference image
        ref_mask (np.ndarray): Reference mask as float [0,1]
        template_image (np.ndarray): Template image to be aligned
        template_mask (np.ndarray): Template mask as bool
        points_per_dim (int): Number of points to evaluate per dimension
        scale_factor (float): Factor to scale bounds by for next recursion
        max_recursions (int): Maximum recursion depth
        current_recursion (int): Current recursion depth
        prev_best_dy (float): Best dy from previous recursion
        prev_best_dx (float): Best dx from previous recursion
        
    Returns:
        tuple: Best shift (delta_y, delta_x)
    """
    # Compute current bounds based on recursion level
    bound_width = 2.0 * (scale_factor ** current_recursion)
    bounds_y = (prev_best_dy - bound_width/2, prev_best_dy + bound_width/2)
    bounds_x = (prev_best_dx - bound_width/2, prev_best_dx + bound_width/2)
    print(f"recursive_mncc_search::bound_width {bound_width}, current_recursion {current_recursion}")
    print(f"recursive_mncc_search::prev_best_dy {prev_best_dy} prev_best_dx {prev_best_dx}")
    print(f"recursive_mncc_search::bounds_y {bounds_y} bounds_x {bounds_x}")

    # Compute best shift for current grid
    best_dy, best_dx, best_score = compute_grid_mncc(
        norm_ref, ref_mask, template_image, template_mask,
        bounds_y, bounds_x, points_per_dim
    )
    print(f"recursive_mncc_search::best_score {best_score}")
    
    # Base case: reached maximum recursions
    if current_recursion >= max_recursions - 1:
        return best_dy, best_dx
    
    # Recursive case: continue search with refined bounds
    return recursive_mncc_search(
        norm_ref, ref_mask, template_image, template_mask,
        points_per_dim, scale_factor, max_recursions,
        current_recursion + 1, best_dy, best_dx
    )

def compute_shift_ncc(ref_image, template_image, ref_mask, template_mask, 
                      points_per_dim=7, max_recursions=10):
    """
    Main function to compute the best shift between reference and template images using MNCC.
    
    Parameters:
        ref_image (np.ndarray): Reference image
        template_image (np.ndarray): Template image to be aligned
        ref_mask (np.ndarray): Reference mask (binary)
        template_mask (np.ndarray): Template mask (binary)
        points_per_dim (int): Number of points to evaluate per dimension
        max_recursions (int): Maximum number of recursive calls
    
    Returns:
        tuple: Best shift (delta_y, delta_x)
    """
    if points_per_dim < 3:
        raise ValueError("points_per_dim must be at least 3")
    if max_recursions < 3:
        raise ValueError("max_recursions must be at least 3")
    
    # Convert masks to float
    ref_mask_float = ref_mask.astype(float)
    
    # Pre-compute normalized reference image statistics
    weight_sum = np.sum(ref_mask_float)
    if weight_sum == 0:
        return 0.0, 0.0
        
    mu_ref = np.sum(ref_image * ref_mask_float) / weight_sum
    sigma_ref = np.sqrt(np.sum(ref_mask_float * (ref_image - mu_ref) ** 2) / weight_sum)
    
    if sigma_ref == 0:
        return 0.0, 0.0
        
    norm_ref = (ref_image - mu_ref) / sigma_ref
    
    # Calculate scale factor based on points_per_dim
    # For example, if points_per_dim = 4, scale_factor ≈ 0.5
    # if points_per_dim = 5, scale_factor ≈ 0.25
    scale_factor = 1.0 / (points_per_dim - 2)
    if scale_factor < 0.25:
        scale_factor = 0.25
    if scale_factor >= 1.0:
        scale_factor = 0.9
    
    # Start recursive search
    return recursive_mncc_search(
        norm_ref, ref_mask_float, template_image, template_mask,
        points_per_dim, scale_factor, max_recursions
    )

def compute_shift_pl(model, ref_image, template_image, ref_mask, template_mask, 
                    points_per_dim=7, max_recursions=10):
    """
    Main function to compute the best shift between reference and template images using
    perceptual loss with VGG features.
    
    Parameters:
        model: VGG-based perceptual loss network
        ref_image (np.ndarray): Reference image
        template_image (np.ndarray): Template image to be aligned
        ref_mask (np.ndarray): Reference mask (binary)
        template_mask (np.ndarray): Template mask (binary)
        points_per_dim (int): Number of points to evaluate per dimension
        max_recursions (int): Maximum number of recursive calls
    
    Returns:
        tuple: Best shift (delta_y, delta_x)
    """
    # Calculate scale factor based on points_per_dim
    scale_factor = 1.0 / (points_per_dim - 1)
    if scale_factor < 0.25:
        scale_factor = 0.25
    if scale_factor >= 1.0:
        scale_factor = 0.9
    
    # Start recursive search
    return recursive_pl_search(
        model, ref_image, ref_mask, template_image, template_mask,
        points_per_dim, scale_factor, max_recursions
    )

def recursive_pl_search(model, ref_image, ref_mask, template_image, template_mask, 
                       points_per_dim, scale_factor, max_recursions, current_recursion=0,
                       prev_best_dy=0.0, prev_best_dx=0.0):
    """
    Recursive function to search for the best alignment using perceptual loss.
    
    Parameters:
        model: VGG-based perceptual loss network
        ref_image (np.ndarray): Reference image
        ref_mask (np.ndarray): Reference mask
        template_image (np.ndarray): Template image to be aligned
        template_mask (np.ndarray): Template mask
        points_per_dim (int): Number of points to evaluate per dimension
        scale_factor (float): Factor to scale bounds by for next recursion
        max_recursions (int): Maximum recursion depth
        current_recursion (int): Current recursion depth
        prev_best_dy (float): Best dy from previous recursion
        prev_best_dx (float): Best dx from previous recursion
        
    Returns:
        tuple: Best shift (delta_y, delta_x)
    """
    # Compute current bounds based on recursion level
    bound_width = 2.0 * (scale_factor ** current_recursion)
    bounds_y = (prev_best_dy - bound_width/2, prev_best_dy + bound_width/2)
    bounds_x = (prev_best_dx - bound_width/2, prev_best_dx + bound_width/2)

    print(f"recursive_pl_search::bound_width {bound_width}, current_recursion {current_recursion}")
    print(f"recursive_pl_search::prev_best_dy {prev_best_dy} prev_best_dx {prev_best_dx}")
    print(f"recursive_pl_search::bounds_y {bounds_y} bounds_x {bounds_x}")
    
    # Compute best shift for current grid
    best_dy, best_dx, best_score = compute_grid_pl(
        model, ref_image, ref_mask, template_image, template_mask,
        bounds_y, bounds_x, points_per_dim
    )
    print(f"recursive_pl_search::best_score {best_score}, best_dy {best_dy}, best_dx {best_dx}")
    # Base case: reached maximum recursions
    if current_recursion >= max_recursions - 1:
        return best_dy, best_dx
    
    # Recursive case: continue search with refined bounds
    return recursive_pl_search(
        model, ref_image, ref_mask, template_image, template_mask,
        points_per_dim, scale_factor, max_recursions,
        current_recursion + 1, best_dy, best_dx
    )

def compute_grid_pl(model, ref_image, ref_mask, template_image, template_mask, 
                   bounds_y, bounds_x, points_per_dim):
    """
    Compute perceptual loss scores for a grid of points within given bounds.
    
    Parameters:
        model: VGG-based perceptual loss network
        ref_image (np.ndarray): Reference image
        ref_mask (np.ndarray): Reference mask
        template_image (np.ndarray): Template image to be aligned
        template_mask (np.ndarray): Template mask
        bounds_y (tuple): (min_y, max_y) bounds for vertical shifts
        bounds_x (tuple): (min_x, max_x) bounds for horizontal shifts
        points_per_dim (int): Number of points to evaluate per dimension
        
    Returns:
        tuple: (best_shift_y, best_shift_x, best_score)
    """
    min_y, max_y = bounds_y
    min_x, max_x = bounds_x
    
    # Generate grid points
    y_points = np.linspace(min_y, max_y, points_per_dim)
    x_points = np.linspace(min_x, max_x, points_per_dim)
    
    best_score = float('inf')  # We want to minimize perceptual loss
    best_shift = (0.0, 0.0)
    
    # Evaluate perceptual loss at each grid point
    for dy in y_points:
        for dx in x_points:
            # Shift template image and mask
            shifted_template, shifted_mask = apply_shift_to_template(dx, dy, template_image, template_mask)
            shifted_mask = (shifted_mask > 0.5).astype(float)
            
            # Compute perceptual loss between reference and shifted template
            pl_score, _ = rm.compute_perceptual_loss(
                ref_image=ref_image,
                mov_image=shifted_template,
                ref_mask=ref_mask,
                mov_mask=shifted_mask,
                model=model
            )
            
            if pl_score < best_score:
                best_score = pl_score
                best_shift = (dy, dx)
    
    return best_shift[0], best_shift[1], best_score

def compute_shift_pcc(ref_image, shifted_image, ref_mask, shifted_mask):
    #i have different way of applying the mask to the image before computing the phase cross correlation
    #for some corner cases where there are a lot of invalid pixels, fully blacking out the masked pixels 
    #results in very wrong results. Weighting them by half seems like it is an ok compromise. Needs more testing though.
    #ref_masked = ref_image * (ref_mask.astype(ref_image.dtype) * 0.1 + 0.9)
    #mov_masked = shifted_image * (shifted_mask.astype(shifted_image.dtype) * 0.1 + 0.9)
    #completely black out the masked pixels
    #ref_masked = ref_image * ref_mask.astype(ref_image.dtype)
    #mov_masked = shifted_image * shifted_mask.astype(shifted_image.dtype)
    #do not mask the pixels at all
    #ref_masked = ref_image 
    #mov_masked = shifted_image

    # Compute the shift (delta_x, delta_y) between reference and template image
    shift_yx, error, diffphase = phase_cross_correlation(
            ref_image,
            shifted_image,
            upsample_factor=1000
        )
    return shift_yx
      

def compute_shift_point_matching(ref_image, tmplt_image, n_keypoints=500, match_threshold=0.75, ransac_threshold=2, scale=4):
    """
    Aligns two images using point matching to estimate a subpixel shift.

    Parameters:
    - ref_image: Reference image (numpy array).
    - tmplt_image: Image to be aligned (numpy array).
    - n_keypoints: Number of keypoints to detect.
    - match_threshold: Threshold for Lowe's ratio test (not directly used here).
    - ransac_threshold: RANSAC inlier threshold in pixels.
    - scale: Scale factor for the images.

    Returns:
    - shift_yx: Estimated (y, x) shift.
    """
    shape = ref_image.shape
    # Convert images to floating point
    image1 = transform.resize(
                img_as_float(ref_image),
                (shape[0]*scale, shape[1]*scale),
                order=3,  # Cubic interpolation
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True)
    

    image2 = transform.resize(
                img_as_float(tmplt_image),
                (shape[0]*scale, shape[1]*scale),
                order=3,  # Cubic interpolation
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True)

    # Initialize ORB detector
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=0.05)

    # Detect and extract features from image1
    orb.detect_and_extract(image1)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    # Detect and extract features from image2
    orb.detect_and_extract(image2)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors

    # Match descriptors using mutual nearest neighbors
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    if len(matches12) < 4:
        raise ValueError("Not enough matches found for reliable alignment.")

    # Extract matched keypoints
    src = keypoints1[matches12[:, 0]][:, ::-1]  # (x, y)
    dst = keypoints2[matches12[:, 1]][:, ::-1]  # (x, y)

    # Estimate translation using RANSAC
    # Define a transformation model: translation only
    model = SimilarityTransform(scale=1, rotation=0, translation=(0, 0))
    try:
        model_robust, inliers = ransac((dst, src),
                                       SimilarityTransform,
                                       min_samples=2,
                                       residual_threshold=ransac_threshold,
                                       max_trials=1000)
    except Exception as e:
        raise ValueError(f"RANSAC failed to find a robust model: {e}")

    if model_robust is None:
        raise ValueError("RANSAC failed to find a robust model.")

    # Extract translation
    estimated_translation = model_robust.translation  # (tx, ty)

    # Since SimilarityTransform includes both x and y translations,
    # and we set rotation=0 and scale=1, we can directly use the translation components.
    shift_x, shift_y = estimated_translation  # (x_shift, y_shift)

    # Apply the estimated shift using scipy.ndimage.shift
    # Note: shift expects (y, x) order
    #aligned_image = ndi_shift(image2, shift=(-shift_y, -shift_x), mode='nearest', order=3)

    # The total estimated shift is (y, x)
    shift_yx = (shift_y/scale, shift_x/scale)
    print(f"point matching shift_yx {shift_yx}")
    return shift_yx

def compute_shift_with_metric(metric_fn, minimize=True, ref_image=None, template_image=None, 
                            ref_mask=None, template_mask=None, points_per_dim=7, max_recursions=10,
                            **metric_kwargs):
    """
    Generic function to compute the best shift between reference and template images using
    any provided metric function.
    
    Parameters:
        metric_fn: Function that computes the similarity/difference metric
                  Should have signature: metric_fn(ref_image, template_image, ref_mask, template_mask, **kwargs)
        minimize: Boolean indicating whether to minimize (True) or maximize (False) the metric
        ref_image (np.ndarray): Reference image
        template_image (np.ndarray): Template image to be aligned
        ref_mask (np.ndarray): Reference mask (binary)
        template_mask (np.ndarray): Template mask (binary)
        points_per_dim (int): Number of points to evaluate per dimension
        max_recursions (int): Maximum number of recursive calls
        **metric_kwargs: Additional keyword arguments to pass to the metric function
    
    Returns:
        tuple: Best shift (delta_y, delta_x)
    """
    # Calculate scale factor based on points_per_dim
    scale_factor = 1.0 / (points_per_dim - 1)
    if scale_factor < 0.25:
        scale_factor = 0.25
    if scale_factor >= 1.0:
        scale_factor = 0.9
    
    # Start recursive search with the generic metric
    return recursive_metric_search(
        metric_fn=metric_fn,
        minimize=minimize,
        ref_image=ref_image,
        ref_mask=ref_mask,
        template_image=template_image,
        template_mask=template_mask,
        points_per_dim=points_per_dim,
        scale_factor=scale_factor,
        max_recursions=max_recursions,
        **metric_kwargs
    )

def recursive_metric_search(metric_fn, minimize, ref_image, ref_mask, template_image, template_mask, 
                          points_per_dim, scale_factor, max_recursions, current_recursion=0,
                          prev_best_dy=0.0, prev_best_dx=0.0, **metric_kwargs):
    """
    Recursive function to search for the best alignment using any metric function.
    
    Parameters:
        metric_fn: Function that computes the similarity/difference metric
        minimize: Boolean indicating whether to minimize or maximize the metric
        ref_image (np.ndarray): Reference image
        ref_mask (np.ndarray): Reference mask
        template_image (np.ndarray): Template image to be aligned
        template_mask (np.ndarray): Template mask
        points_per_dim (int): Number of points to evaluate per dimension
        scale_factor (float): Factor to scale bounds by for next recursion
        max_recursions (int): Maximum recursion depth
        current_recursion (int): Current recursion depth
        prev_best_dy (float): Best dy from previous recursion
        prev_best_dx (float): Best dx from previous recursion
        **metric_kwargs: Additional arguments to pass to metric_fn
        
    Returns:
        tuple: Best shift (delta_y, delta_x)
    """
    # Compute current bounds based on recursion level
    bound_width = 2.0 * (scale_factor ** current_recursion)
    bounds_y = (prev_best_dy - bound_width/2, prev_best_dy + bound_width/2)
    bounds_x = (prev_best_dx - bound_width/2, prev_best_dx + bound_width/2)

    print(f"recursive_metric_search::bound_width {bound_width}, current_recursion {current_recursion}")
    print(f"recursive_metric_search::prev_best_dy {prev_best_dy} prev_best_dx {prev_best_dx}")
    print(f"recursive_metric_search::bounds_y {bounds_y} bounds_x {bounds_x}")
    
    # Compute best shift for current grid
    best_dy, best_dx, best_score = compute_grid_metric(
        metric_fn=metric_fn,
        minimize=minimize,
        ref_image=ref_image,
        ref_mask=ref_mask,
        template_image=template_image,
        template_mask=template_mask,
        bounds_y=bounds_y,
        bounds_x=bounds_x,
        points_per_dim=points_per_dim,
        **metric_kwargs
    )
    
    print(f"recursive_metric_search::best_score {best_score}, best_dy {best_dy}, best_dx {best_dx}")
    
    # Base case: reached maximum recursions
    if current_recursion >= max_recursions - 1:
        return best_dy, best_dx
    
    # Recursive case: continue search with refined bounds
    return recursive_metric_search(
        metric_fn=metric_fn,
        minimize=minimize,
        ref_image=ref_image,
        ref_mask=ref_mask,
        template_image=template_image,
        template_mask=template_mask,
        points_per_dim=points_per_dim,
        scale_factor=scale_factor,
        max_recursions=max_recursions,
        current_recursion=current_recursion + 1,
        prev_best_dy=best_dy,
        prev_best_dx=best_dx,
        **metric_kwargs
    )

def compute_grid_metric(metric_fn, minimize, ref_image, ref_mask, template_image, template_mask, 
                       bounds_y, bounds_x, points_per_dim, **metric_kwargs):
    """
    Compute metric scores for a grid of points within given bounds.
    
    Parameters:
        metric_fn: Function that computes the similarity/difference metric
        minimize: Boolean indicating whether to minimize or maximize the metric
        ref_image (np.ndarray): Reference image
        ref_mask (np.ndarray): Reference mask
        template_image (np.ndarray): Template image to be aligned
        template_mask (np.ndarray): Template mask
        bounds_y (tuple): (min_y, max_y) bounds for vertical shifts
        bounds_x (tuple): (min_x, max_x) bounds for horizontal shifts
        points_per_dim (int): Number of points to evaluate per dimension
        **metric_kwargs: Additional arguments to pass to metric_fn
        
    Returns:
        tuple: (best_shift_y, best_shift_x, best_score)
    """
    min_y, max_y = bounds_y
    min_x, max_x = bounds_x
    
    # Generate grid points
    y_points = np.linspace(min_y, max_y, points_per_dim)
    x_points = np.linspace(min_x, max_x, points_per_dim)
    
    best_score = float('inf') if minimize else float('-inf')
    best_shift = (0.0, 0.0)
    
    # Evaluate metric at each grid point
    for dy in y_points:
        for dx in x_points:
            # Shift template image and mask
            shifted_template, shifted_mask = apply_shift_to_template(dx, dy, template_image, template_mask)
            shifted_mask = (shifted_mask > 0.5).astype(float)
            
            # Compute metric between reference and shifted template
            score = metric_fn(
                ref_image=ref_image,
                mov_image=shifted_template,
                ref_mask=ref_mask,
                mov_mask=shifted_mask,
                **metric_kwargs
            )
            
            # Update best score based on minimization/maximization
            if (minimize and score < best_score) or (not minimize and score > best_score):
                best_score = score
                best_shift = (dy, dx)
    
    return best_shift[0], best_shift[1], best_score

