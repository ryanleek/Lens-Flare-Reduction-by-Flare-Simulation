# Lens-Flare-Reduction-by-Flare-Simulation
Lens Flare Reduction by Flare Simulation and U-Net

<p float="left">
  <img src="/images/origin.jpg" width="200" />
  <img src="/images/mask.jpg" width="200" /> 
  <img src="/images/output.jpg" width="200" /> 
</p>

1. Prepare Image Set (Image Set Used in Training: https://www.kaggle.com/arnaud58/landscape-pictures)
2. Apply Flare Simulation to Image Set using flare_simulation.py
3. Create Dataloader using dataloader.py
4. Run train.py to start training
5. Run predict.py to predict flare area and inpaint flared image

*.ipynb colab environment
*Training structure inspired by: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55
