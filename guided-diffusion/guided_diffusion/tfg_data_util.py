import torch
import cv2
import numpy as np

def normalise2(tensor):
    '''[0,1] -> [-1,1]'''
    return (tensor*2 - 1.).clamp(-1,1)

def tfg_data(dataloader, face_hide_percentage, use_ref, use_audio):#, sampling_use_gt_for_ref=False, noise = None):
    def inf_gen(generator):
        while True:
            yield from generator
    data = inf_gen(dataloader)
    for batch in data:
        img_batch, model_kwargs = tfg_process_batch(batch, face_hide_percentage, use_ref, use_audio)
        yield img_batch, model_kwargs
        

def tfg_process_batch(batch, face_hide_percentage, use_ref=False, use_audio=False, sampling_use_gt_for_ref=False, noise=None,
                     use_gradient_mask=False, blur_kernel_size=15, use_circular_mask=False, mask_shape='ellipse', ellipse_aspect_ratio=1.5):
    """
    Process batch with optional gradient masking support
    
    Args:
        batch: Input batch data
        face_hide_percentage: Portion of face to regenerate
        use_ref: Whether to use reference image
        use_audio: Whether to use audio conditioning
        sampling_use_gt_for_ref: Whether to use ground truth for reference
        noise: Optional noise tensor
        use_gradient_mask: Whether to use gradient masking (Phase 1.1)
        blur_kernel_size: Gaussian blur size for gradient masking
    
    Returns:
        img_batch: Processed image batch
        model_kwargs: Model keyword arguments
    """
    model_kwargs = {}
    B, F,C, H, W = batch["image"].shape
    img_batch = normalise2(batch["image"].reshape(B*F, C, H, W).contiguous())
    model_kwargs = tfg_add_cond_inputs(img_batch, model_kwargs, face_hide_percentage, noise,
                                     use_gradient_mask, blur_kernel_size, use_circular_mask, mask_shape, ellipse_aspect_ratio)
    if use_ref:
        model_kwargs = tfg_add_reference(batch, model_kwargs, sampling_use_gt_for_ref)
    if use_audio:
        model_kwargs = tfg_add_audio(batch,model_kwargs)
    return img_batch, model_kwargs

def tfg_add_reference(batch, model_kwargs, sampling_use_gt_for_ref=False):
    # assuming nrefer = 1
    #[B, nframes, C, H, W] -> #[B*nframes, C, H, W]
    if sampling_use_gt_for_ref:
        B, F,C, H, W = batch["image"].shape
        img_batch = normalise2(batch["image"].reshape(B*F, C, H, W).contiguous())
        model_kwargs["ref_img"] = img_batch
    else:
        _, _, C, H , W =  batch["ref_img"].shape
        ref_img = normalise2(batch["ref_img"].reshape(-1, C, H, W).contiguous())
        model_kwargs["ref_img"] = ref_img
    return model_kwargs

def tfg_add_audio(batch, model_kwargs):
    # unet needs [BF, h, w] as input
    B, F, _, h, w = batch["indiv_mels"].shape
    indiv_mels = batch["indiv_mels"] # [B, F, 1, h, w]
    indiv_mels = indiv_mels.squeeze(dim=2).reshape(B*F, h , w)
    model_kwargs["indiv_mels"] = indiv_mels
    # syncloss needs [B, 1, 80, 16] as input
    if "mel" in batch:
        mel = batch["mel"] #[B, 1, h, w]
        model_kwargs["mel"]=mel
    return model_kwargs

def create_gradient_mask(B, H, W, face_hide_percentage, blur_kernel_size, device, dtype):
    """
    Create gradient mask for smoother blending - Phase 1.1 enhancement
    
    Args:
        B: Batch size
        H: Height
        W: Width  
        face_hide_percentage: Portion of face to regenerate (0.5 = bottom 50%)
        blur_kernel_size: Gaussian blur kernel size for smooth transitions
        device: Torch device
        dtype: Torch data type
    
    Returns:
        mask: Gradient mask tensor [B, 1, H, W]
    """
    mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    mask_start_idx = int(H * (1 - face_hide_percentage))
    
    if blur_kernel_size > 0:
        # REVISED APPROACH: Use controlled Gaussian blur that only affects the transition region
        # This preserves the natural blur characteristics while preventing top region contamination
        
        # Start with hard mask
        mask[:, :, mask_start_idx:, :] = 1.0
        
        # Only blur the transition region, not the entire mask
        transition_width = min(blur_kernel_size, H - mask_start_idx - 5)  # Leave some bottom margin
        
        if transition_width > 2:  # Only apply if we have enough space
            # Convert to numpy for cv2 operations (only the transition region)
            mask_np = mask.cpu().numpy()
            
            for b in range(B):
                # Extract only the transition region for blurring
                transition_start = mask_start_idx - blur_kernel_size // 2
                transition_end = mask_start_idx + blur_kernel_size
                
                # Ensure boundaries are within image
                transition_start = max(0, transition_start)
                transition_end = min(H, transition_end)
                
                if transition_end > transition_start:
                    # Create a temporary mask for just the transition region
                    transition_mask = mask_np[b, 0, transition_start:transition_end, :].copy()
                    
                    # Apply Gaussian blur only to this region
                    blurred_transition = cv2.GaussianBlur(
                        transition_mask, 
                        (blur_kernel_size, blur_kernel_size), 
                        0
                    )
                    
                    # Put the blurred transition back, but preserve top region
                    mask_np[b, 0, transition_start:transition_end, :] = blurred_transition
                    
                    # CRITICAL: Ensure top region remains exactly zero
                    mask_np[b, 0, :mask_start_idx, :] = 0.0
            
            # Convert back to tensor and ensure proper bounds
            mask = torch.from_numpy(mask_np).to(device=device, dtype=dtype).clamp(0, 1)
        
    else:
        # Hard masking (original behavior)
        mask[:, :, mask_start_idx:, :] = 1.0
    
    return mask

def create_circular_mask(B, H, W, face_hide_percentage, mask_shape='ellipse', ellipse_aspect_ratio=1.5, device=None, dtype=None):
    """
    Create circular or elliptical mask for more natural face masking - Phase 1.4 enhancement
    
    Args:
        B: Batch size
        H: Height
        W: Width
        face_hide_percentage: Portion of face to regenerate (0.5 = bottom 50%)
        mask_shape: 'rectangle', 'circle', or 'ellipse'
        ellipse_aspect_ratio: Width/height ratio for elliptical masks (1.0 = circle)
        device: Torch device
        dtype: Torch data type
    
    Returns:
        mask: Circular/elliptical mask tensor [B, 1, H, W]
    """
    mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    
    if mask_shape == 'rectangle':
        # Original rectangular masking
        mask_start_idx = int(H * (1 - face_hide_percentage))
        mask[:, :, mask_start_idx:, :] = 1.0
    
    elif mask_shape in ['circle', 'ellipse']:
        # Create circular or elliptical mask
        center_y = H * (1 - face_hide_percentage / 2)  # Center the mask in the face region
        center_x = W / 2
        
        if mask_shape == 'circle':
            # Circle: use smaller dimension to ensure it fits
            radius = min(H * face_hide_percentage / 2, W / 2)
            radius_y = radius_x = radius
        else:  # ellipse
            # Ellipse: adjust radii based on aspect ratio
            radius_y = H * face_hide_percentage / 2
            radius_x = radius_y * ellipse_aspect_ratio
            # Ensure ellipse fits within image bounds
            radius_x = min(radius_x, W / 2)
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=dtype).view(-1, 1).expand(H, W)
        x_coords = torch.arange(W, device=device, dtype=dtype).view(1, -1).expand(H, W)
        
        # Calculate ellipse equation: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
        ellipse_mask = ((x_coords - center_x) / radius_x) ** 2 + ((y_coords - center_y) / radius_y) ** 2 <= 1
        
        # Convert to float and expand to batch dimensions
        ellipse_mask = ellipse_mask.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        mask = ellipse_mask
    
    return mask

def tfg_add_cond_inputs(img_batch, model_kwargs, face_hide_percentage, noise=None, 
                       use_gradient_mask=False, blur_kernel_size=15, use_circular_mask=False, 
                       mask_shape='ellipse', ellipse_aspect_ratio=1.5):
    """
    Add conditional inputs with optional gradient masking
    
    Args:
        img_batch: Input image batch
        model_kwargs: Model keyword arguments
        face_hide_percentage: Portion of face to regenerate
        noise: Noise tensor (optional)
        use_gradient_mask: Whether to use gradient masking (Phase 1.1)
        blur_kernel_size: Gaussian blur size for gradient masking
    
    Returns:
        model_kwargs: Updated model kwargs with mask and conditional image
    """
    B, C, H, W = img_batch.shape
    
    if use_circular_mask:
        # Use circular/elliptical masking - Phase 1.4 enhancement
        mask = create_circular_mask(B, H, W, face_hide_percentage, mask_shape, ellipse_aspect_ratio,
                                  img_batch.device, img_batch.dtype)
    elif use_gradient_mask and blur_kernel_size > 0:
        # Use gradient masking - Phase 1.1 enhancement
        mask = create_gradient_mask(B, H, W, face_hide_percentage, blur_kernel_size, 
                                  img_batch.device, img_batch.dtype)
    else:
        # Original hard rectangular masking
        mask = torch.zeros(B,1,H,W, device=img_batch.device, dtype=img_batch.dtype)
        mask_start_idx = int (H*(1-face_hide_percentage))
        mask[:,:,mask_start_idx:,:]=1.
    
    if noise is None:
        noise = torch.randn_like(img_batch)
    assert noise.shape == img_batch.shape, "Noise shape != Image shape"
    cond_img = img_batch *(1. - mask)+mask*noise

    model_kwargs["cond_img"] = cond_img
    model_kwargs["mask"] = mask
    return model_kwargs


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn=nn*s
        pp+=nn
    return pp