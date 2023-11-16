# Radial Sampling in MRI 

My journey in the field of medical imaging began with this project. Magnetic resonance imaging is a powerful tool for obtaining images of soft body tissues. 
In magnetic resonance imaging, the signal is acquired in the so-called k-space (Fourier space) and subsequently converted into the spatial domain of the image. If sufficient samples are acquired to satisfy the Nyquist theorem, then the reconstructed image does not lose information compared to the original. However, acquiring an entire Cartesian plane is time-consuming, leading to issues such as prolonged machine usage for a single patient, motion artifacts, and patient discomfort. In the figure its possible to see how a slice of the image appear in the Fourier space and in the image domain. 

<img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/Central slice - cartesian kspace.png" alt="Testo alternativo" width="45%"><img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/Central slide.png" alt="Testo alternativo" width="50%">

In this project, we analyze techniques for reconstructing images from radial sampling of the k-space (or Fourier space). 

<img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/Trajectories - Gold angle Tight.png" alt="Testo alternativo" width="50%">

In particular, we focus on the inverse fast Fourier transform (the technique used for cartesian sampling), non-uniform adjoint fast Fourier transform, wavelet transform, and total variation reconstruction. To compare the image reconstructed respecting the Nyquist theorem with images reconstructed from radial sampling, various quantitative metrics (such as cumulative error) and qualitative metrics (intensity difference, gradient difference) were employed.

The best part of the project, in my opinion, was the application of the image reconstruction method used in computerized tomography, called **filtered backprojection**, to the reconstruction of MRI images (as it is possible to see in the figure). 

<img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/CT_differentR.jpg" alt="Testo alternativo" width="50%">

This step required a deep understanding of the mechanisms of two fundamental techniques in medicine, CT and MRI. It was truly satisfying to be able to accomplish this!





This project was carried out as part of the 3D Image Reconstruction and Analysis in Medicine course at KTH-Royal Institute of Technology. The project was conducted in collaboration with another student, Ana Candela."

[Return to initial page](https://github.com/RebeccaBonato/Master-Projects-/blob/main/README.md)
