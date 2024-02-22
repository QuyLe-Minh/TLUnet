from architecture.TLUnet import TLUnet

model = TLUnet(n_classes=1)
# model.apply(init_weights)
n = 0
for (name,param) in model.named_parameters():
  if n >= 37: break
  print(name)
  n += 1
