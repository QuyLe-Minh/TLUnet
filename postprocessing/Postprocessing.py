import torch
from scipy import ndimage

class Postprocessing:
    def __init__(self):
        pass
        
    def __call__(self, pred):
        """Postprocessing prediction

        Args:
            pred (torch tensor cuda): B = 1, C = 1, H, W, D

        Returns:
            torch tensor cuda 1, 1, H, W, D
        """
        mask = pred.reshape(pred.shape[-3:]).cpu() #H, W, D
        h, w, d = mask.shape
        for i in range(mask.shape[-1]):
            slice = mask[:, :, i]
            slice = self.denoising(slice)
            slice = self.fill_hole(slice)
            mask[:, :, i] = slice
        
        mask = mask.reshape(1, 1, h, w, d).cuda()
        return mask
        
    def labeling(self, mask):
        """Labelling continuous component

        Args:
            mask (numpy arr): Liver mask
        """
        m, n = mask.shape
        label = 0
        res = torch.zeros(mask.shape)
        di = [-1, 0, 1, 0]
        dj = [0, 1, 0, -1]
        q = []
        for i in range(m):
            for j in range(n):
                if mask[i,j] == 1 and res[i, j] == 0:
                    label+=1
                    res[i, j] = label
                    q.append((i,j))
                else:
                    continue
                while q:
                    r, c = q.pop(0)
                    for k in range(4):
                            nr, nc = r + di[k], c+dj[k]
                            if nr >= m or nr < 0 or nc >= n or nc < 0 or res[nr, nc] != 0:
                                continue
                            if mask[nr, nc] == 1:
                                res[nr, nc] = label
                                q.append((nr, nc))

        return res, label                              

    def fill_hole(self, mask):
        mask = ndimage.binary_fill_holes(mask).astype(int) 
        return torch.tensor(mask) 

    def denoising(self, mask):
        label, mx = -1, -1
        res, n = self.labeling(mask)
        for l in range(1, n+1):
            curr = 0
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if res[i, j] == l:
                        curr+=1
            if curr > mx:
                mx = curr
                label = l
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if res[i, j] != label:
                    mask[i, j] = 0
        return mask
                
    

# mask = [[1,1,1,1,1,1],
#         [1,0,0,1,0,0],
#         [1,1,1,1,0,0],
#         [1,1,1,1,0,0],
#         [1,1,0,1,0,0]]
# mask = np.array(mask)
# mask = Postprocessing()(mask)
# print(mask)    