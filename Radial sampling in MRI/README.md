# Radial Sampling in MRI 

My journey in the field of medical imaging began with this project. Magnetic resonance imaging is a powerful tool for obtaining images of soft body tissues. 
In magnetic resonance imaging, the signal is acquired in the so-called k-space (Fourier space) and subsequently converted into the spatial domain of the image. If sufficient samples are acquired to satisfy the Nyquist theorem, then the reconstructed image does not lose information compared to the original. However, acquiring an entire Cartesian plane is time-consuming, leading to issues such as prolonged machine usage for a single patient, motion artifacts, and patient discomfort. 

<img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/Central slice - cartesian kspace.png" alt="Testo alternativo" width="42%"><img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/Central slide.png" alt="Testo alternativo" width="50%">

In this project, we analyze techniques for reconstructing images from radial sampling of the k-space (or Fourier space). 

<img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/Trajectories - Gold angle Tight.png" alt="Testo alternativo" width="80%">

In particular, we focus on the inverse fast Fourier transform (the technique used for cartesian sampling), non-uniform adjoint fast Fourier transform, wavelet transform, and total variation reconstruction. To compare the image reconstructed respecting the Nyquist theorem with images reconstructed from radial sampling, various quantitative metrics (such as cumulative error) and qualitative metrics (intensity difference, gradient difference) were employed. Here the [code](https://github.com/RebeccaBonato/Master-Projects-/blob/main/Radial%20sampling%20in%20MRI/Project_final.ipynb) of our investigation and assessment. 

The best part of the project, in my opinion, was the application of the image reconstruction method used in computerized tomography, called **FILTERED BACKPROJECTION**, to the reconstruction of MRI images (here the [code](https://github.com/RebeccaBonato/Master-Projects-/blob/main/Radial%20sampling%20in%20MRI/CT%20Reconstruction.ipynb) of this task and below you can see results of our reconstruction). 

<img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/CT_differentR.jpg" alt="Testo alternativo" width="80%">

This step required a deep understanding of the mechanisms of two fundamental imaging techniques in medicine, CT and MRI. It was truly satisfying to be able to accomplish this!!

**If you want to know more about it**, here link to the [project report](https://github.com/RebeccaBonato/Master-Projects-/blob/main/Radial%20sampling%20in%20MRI/Project_1%20-%20Report_Project_1.pdf).

## :handshake: Contributors
The project was carried out in collaboration Ana Candela Celdrán as part of the 3D Image Reconstruction and Analysis in Medicine course at KTH-Royal Institute of Technology. 

[Return to initial page](https://github.com/RebeccaBonato/Master-Projects-/blob/main/README.md)
