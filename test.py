from src.models import fullnet
from src.utils import utils as u

""" model = fullnet.FullNetWrapper()
model.train('18', epochs=100)
model.save('src/experiments/fullnet/experiment_0/models', 'test.tar') """
model = fullnet.FullNetWrapper('src/experiments/fullnet/experiment_0/models/test.pth')

master, *(_) = u.extract_central_patch(u.extract_s2mat('data/images/mat/S2/18.mat'), 1024)
slave, *(_) = u.extract_central_patch(u.extract_s3mat('data/images/mat/S3/18.mat'), 1024//15)

model.evaluate(master, slave)