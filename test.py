from src.models.models import *
from src.utils import utils as u
import os

""" model = fullnet.FullNetWrapper()
model.train('18', epochs=100)
model.save('src/experiments/fullnet/experiment_0/models', 'test.tar') """
#model = FullNetWrapper('src/experiments/fullnet/experiment_0/models/test.pth')
routes = u.get_files_path('data/images/S3')

for i in range(5):
    samples = 2**i
    chosen = np.random.choice(routes, samples, replace=False)
    images = np.array([u.extract_s3mat(route) for route in chosen])
    images = np.moveaxis(images, -1, 1)
    model = RuNetv2()
    model.train(images, epochs=1000, learning_rate=0.0001, upsamplings=2)
    model.save('src/experiments/runet')

    
#model.train()
#master, *(_) = u.extract_central_patch(u.extract_s2mat('data/images/mat/S2/18.mat'), 1024)
#slave, *(_) = u.extract_central_patch(u.extract_s3mat('data/images/mat/S3/18.mat'), 1024//15)

#model.evaluate(master, slave)