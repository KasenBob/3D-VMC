import torch
from src.model.losses import DiceLoss


def compute_iou(pred, gt):
    pred = pred.clone()
    pred[pred <= 0.5] = 0
    pred[pred >= 0.5] = 1
    intersection = torch.sum(pred.mul(gt)).float()
    union = torch.sum(torch.ge(pred.add(gt), 1)).float()
    return intersection / union


def compute_iou_score(pred, gt):
    socre_list = []
    for i, j in zip(pred, gt):
        score = compute_iou(i, j)
        socre_list.append(score)
    batch_score = torch.stack(socre_list)
    return batch_score


def generate_indices(side, device=torch.device('cpu')):
    """ Generates meshgrid indices. 
        May be helpful to cache when converting lots of shapes.
    """
    r = torch.arange(0, side+2).to(device)
    id1,id2,id3 = torch.meshgrid([r,r,r])
    return id1.short(), id2.short(), id3.short()


def encode_shapelayer(voxel, id1=32, id2=32, id3=32):
    """ Encodes a voxel grid into a shape layer
        by projecting the enclosed shape to 6 depth maps.
        Returns the shape layer and a reconstructed shape.
        The reconstructed shape can be used to recursively encode
        complex shapes into multiple layers.
        Optional parameters id1,id2,id3 can save memory when multiple
        shapes are to be encoded. They can be constructed like this:
        r = np.arange(0,voxel.shape[0]+2)
        id1,id2,id3 = np.meshgrid(r,r,r, indexing='ij')
    """
    
    device = voxel.device

    side = voxel.shape[0]
    assert voxel.shape[0] == voxel.shape[1] and voxel.shape[1] == voxel.shape[2], \
        'The voxel grid needs to be a cube. It is however %dx%dx%d.' % \
        (voxel.shape[0],voxel.shape[1],voxel.shape[2])

    if id1 is None or id2 is None or id3 is None:
        id1, id2, id3 = generate_indices(side, device)
        pass

    # add empty border for argmax
    # need to distinguish empty tubes
    v = torch.zeros(side+2,side+2,side+2, dtype=torch.uint8, device=device)
    v[1:-1,1:-1,1:-1] = voxel
        
    shape_layer = torch.zeros(side, side, 6, dtype=torch.int16, device=device)
    
    # project depth to yz-plane towards negative x
    s1 = torch.argmax(v, dim=0) # returns first occurence
    # project depth to yz-plane towards positive x
    # s2 = torch.argmax(v[-1::-1,:,:], dim=0) #, but negative steps not supported yet
    s2 = torch.argmax(torch.flip(v, [0]), dim=0)
    s2 = side+1-s2 # correct for added border

    # set all empty tubes to 0    
    s1[s1 < 1] = side+2
    s2[s2 > side] = 0    
    shape_layer[:,:,0] = s1[1:-1,1:-1]
    shape_layer[:,:,1] = s2[1:-1,1:-1]
    
    # project depth to xz-plane towards negative y
    s1 = torch.argmax(v, dim=1)
    # project depth to xz-plane towards positive y
    # s2 = torch.argmax(v[:,-1::-1,:], dim=1) #, but negative steps not supported yet
    s2 = torch.argmax(torch.flip(v, [1]), dim=1)
    s2 = side+1-s2

    s1[s1 < 1] = side+2
    s2[s2 > side] = 0    
    shape_layer[:,:,2] = s1[1:-1,1:-1]
    shape_layer[:,:,3] = s2[1:-1,1:-1]
    
    # project depth to xy-plane towards negative z
    s1 = torch.argmax(v, dim=2)
    # project depth to xy-plane towards positive z
    # s2 = torch.argmax(v[:,:,-1::-1], dim=2) #, but negative steps not supported yet
    s2 = torch.argmax(torch.flip(v, [2]), dim=2)
    s2 = side+1-s2

    s1[s1 < 1] = side+2
    s2[s2 > side] = 0    
    shape_layer[:,:,4] = s1[1:-1,1:-1]
    shape_layer[:,:,5] = s2[1:-1,1:-1]
    
    return shape_layer.float()


def pos_loss(pred, target, num_components=6):
    """ Modified L1-loss, which penalizes background pixels
        only if predictions are closer than 1 to being considered foreground.
    """

    fg_loss  = pred.new_zeros(1)
    bg_loss  = pred.new_zeros(1)
    fg_count = 0 # counter for normalization
    bg_count = 0 # counter for normalization

    for i in range(num_components):
        mask     = target[:, i, :, :].gt(0).float().detach()
        target_i = target[:, i, :, :]
        pred_i   = pred[:, i, :, :]

        # L1 between prediction and target only for foreground
        fg_loss  += torch.mean((torch.abs(pred_i-target_i)).mul(mask))
        fg_count += torch.mean(mask)

        # flip mask => background
        mask = 1-mask

        # L1 for background pixels > -1
        bg_loss  += torch.mean(((pred_i + 1)).clamp(min=0).mul(mask))
        bg_count += torch.mean(mask)
        pass

    return fg_loss / max(1, fg_count) + \
           bg_loss / max(1, bg_count)


def shl2shlx(shape_layers):
    """ Modify shape layer representation for better learning.
        1. We reflect each odd depth map => background is always zeros, shape always positive depth.
        2. We transpose the first two depth maps => structures of shapes align better across depth maps.
    """
    side = shape_layers.shape[-1]
    shape_layers[::2,:,:] = side+2 - shape_layers[::2,:,:]
    shape_layers[:2,:,:]  = shape_layers[:2,:,:].clone().permute(0,2,1)
    return shape_layers


def compute_depth_score(pred, gt):
    pred_list = []
    gt_list = []

    for i, j in zip(pred, gt):

        x_d = encode_shapelayer(i)
        x_d = x_d.swapaxes(2, 1)
        x_d = x_d.swapaxes(1, 0)
        x_d = shl2shlx(x_d)
        pred_list.append(x_d)

        y_d = encode_shapelayer(j)
        y_d = y_d.swapaxes(2, 1)
        y_d = y_d.swapaxes(1, 0)
        y_d = shl2shlx(y_d)
        gt_list.append(y_d)
    
    pred = torch.stack(pred_list)
    gt = torch.stack(gt_list)
    # all -> [3, 6, 32, 32]

    loss = pos_loss(pred, gt)

    return loss


    
        

        
        
        