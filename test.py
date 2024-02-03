from architecture.TLUnet import TLUnet

model = TLUnet(n_classes=2)
# model.apply(init_weights)
n = 0
for param in model.named_parameters():
  if n < 37:
    print(param, n)
    n+=1
  else: break
