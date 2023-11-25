# femur_segmentation_atlas
The aim of the project consistis in devoloping useful strategies for hip surgery planning starting from 3D CT images. In particular it focuses on atlas based segmentation of the left femur and pelvis. 
Atlas based segmentation is based on registration that is carried out in a linear and non-linear fashion. 

The project consists of:
 - Manual segmentation of the femoral head and pelvis using seeded region growing in 3D Slicer. This has been done on three atlas images.

<p align="center">
  <img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/3Dslicer_3view.png" alt="3D slicer" width="40%">
  <img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/3Dslicer.png" alt="3D slicer segmented" width="33%">
</p>

 - Registration of atlas images to target image using SITK
    - First linear registration (Similarity transform)
    - Then non-linear registration (B-Spline)
 - Segmenting target image based on the agreement of registered atlas segmentations using majority voting.

<p align="center">
  <img src="https://github.com/RebeccaBonato/Master-Projects-/blob/main/images/Atlas_Segmentation.png" alt="Atlas based segmentation" width="70%">
</p>
 
 Furthermore, this repository includes implementations to evaluate the segmentation results using Hausdorff distance and DSC. Notice that the function is specific to this project, were the labels of interest had the value 2 and 4. 
 
 In addition, a there is a Jupyter Notebook, were a pipeline for Obturator Foramen detection was implemented. The used network was too complex for the data at hand. To achieve successful identification of the Obturator Foramen, it is necessary to expand the training dataset and to resample images to have isotropic pixels. Another strategy could be to choose a simpler network with fewer parameters to train. 
 
 The report is structured in a way that the [main.py file](main.py) starts defines the atlas images as a list of paths and also defines images that are supposed to be segmented. The files [segmentation](segmentation.py), [registration](registration.py), [transformation](transformation.py) and [assessment](assessment.py) implement their respective functions. For more details about the project see [project_report.pdf](project_report.pdf).


 ## :handshake: Contributors
 This project has been carried out in collaboration with Lasse Stahnke as a part of the course 3D reconstruction and analysis in images at KTH-Royal Institute of Technology. 

[Return to initial page](https://github.com/RebeccaBonato/Master-Projects-/blob/main/README.md)
