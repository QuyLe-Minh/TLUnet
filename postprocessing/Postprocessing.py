import numpy as np
from scipy import ndimage

class Postprocessing:
    def __init__(self, mask):
        self.mask = mask
        
    def __call__(self):
        return self.run()
        
    def labeling(self):
        """Labelling continuous component

        Args:
            mask (numpy arr): Liver mask
        """
        m, n = self.mask.shape
        label = 0
        res = np.zeros(self.mask.shape)
        di = [-1, 0, 1, 0]
        dj = [0, 1, 0, -1]
        q = []
        for i in range(m):
            for j in range(n):
                if self.mask[i,j] == 1 and res[i, j] == 0:
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
                            if self.mask[nr, nc] == 1:
                                res[nr, nc] = label
                                q.append((nr, nc))

        return res, label                              

    def fill_hole(self):
        mask = ndimage.binary_fill_holes(self.mask).astype(int) 
        return mask  

    def denoising(self):
        label, mx = -1, -1
        res, n = self.labeling()
        for l in range(1, n+1):
            curr = 0
            for i in range(self.mask.shape[0]):
                for j in range(self.mask.shape[1]):
                    if res[i, j] == l:
                        curr+=1
            if curr > mx:
                mx = curr
                label = l
        
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if res[i, j] != label:
                    self.mask[i, j] = 0
    
    def run(self):
        self.denoising()
        mask = self.fill_hole()
        return mask
                
    

# mask = [[1,1,1,1,1,1],
#         [1,0,0,1,0,0],
#         [1,1,1,1,0,0],
#         [1,1,1,1,0,0],
#         [1,1,0,1,0,0]]
# mask = np.array(mask)
# post = Postprocessing(mask)
# mask = post()
# print(mask)    