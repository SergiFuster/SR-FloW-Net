from src.models.models import *
from src.utils import utils as u

""" model = fullnet.FullNetWrapper()
model.train('18', epochs=100)
model.save('src/experiments/fullnet/experiment_0/models', 'test.tar') """
#model = FullNetWrapper('src/experiments/fullnet/experiment_0/models/test.pth')

model = RuNet()
images = np.array([u.extract_s3mat(''), u.extract_s3mat('')])
print(images.shape)
#model.train()
#master, *(_) = u.extract_central_patch(u.extract_s2mat('data/images/mat/S2/18.mat'), 1024)
#slave, *(_) = u.extract_central_patch(u.extract_s3mat('data/images/mat/S3/18.mat'), 1024//15)

#model.evaluate(master, slave)