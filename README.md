# IH-GAN_CMAME_2022
Data and code associated with our accepted CMAME 2022 paper: [IH-GAN: A Conditional Generative Model for Implicit Surface-Based Inverse Design of Cellular Structures](https://arxiv.org/pdf/2103.02588.pdf).

![Alt text](ihgan.png)

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Wang, J., Chen, W., Da, D., Fuge, M., & Rai, R. (2021). IH-GAN: A Conditional Generative Model for Implicit Surface-Based Inverse Design of Cellular Structures. arXiv preprint arXiv:2103.02588.

	@article{wang2021ih,
      title={IH-GAN: A Conditional Generative Model for Implicit Surface-Based Inverse Design of Cellular Structures},
      author={Wang, Jun and Chen, Wei and Da, Daicong and Fuge, Mark and Rai, Rahul},
      journal={Computer Methods in Applied Mechanics and Engineering},
      year={2022}
    }

## Dataset

The dataset can be downloaded from [Google Drive](link by Jun). It involves the shape parameters and the effective material properties of ~900 unit cells. Please check the paper for detailed information on the dataset.

![Alt text](data.png)

## Code Usage

1. (Jun) Run Matlab code ... to obtain the optimal material property distribution. The optimal material properties will be saved at `opt/tgt_prp.mat`.

2. Create a conda environment called `ihgan`:

   ```
   conda env create -f environment.yml
   ```
   
3. Train the IH-GAN model and generate unit cell shapes based on the optimal material properties:

   ```
   python gen_shape.py train
   ```

   The trained model will be saved in the folder `trained_model/`. 
   
   The generated unit cell shape parameters will be saved at `opt/dvar_synth.npy` and `opt/dvar_synth.mat`.
   
4. (Jun) The unit cell shapes can be assembled to form an optimized metamaterial system ...