.. _ScanHarm_Overview:

=======================
Data harmonization for neuroscientific research: Theory, challenges, and applications.
=======================


Overview
--------

This course provides a comprehensive, hands-on journey from understanding the fundamental challenges of site effects in neuroimaging data to implementing state-of-the-art harmonization techniques.

This course is based on the **OHBM 2026 Educational Course on "Data harmonization for neuroscientific research: Theory, challenges, and applications,"** originally presented at the OHBM 2026 annual meeting. 
Developed by `Nicolás Nieto <https://github.com/N-Nieto>`_ [✉️](n.nieto@fz-juelich.de), `Johanna Bayer <https://github.com/likeajumprope>`_, `Gaurav Bhalerao <https://github.com/gvbhalerao591>`_, `Emma Prevot <https://github.com/emmaprevot>`_, and `Jacob Turnbull <https://github.com/Jake-Turnbull>`_, 
this material provides comprehensive hands-on instruction for understanding and implementing harmonization techniques in neuroimaging. For the complete course materials and code, please visit the `original repository <https://github.com/N-Nieto/OHBM2026_Educational_course_harmonization/tree/main>`_.

Learning Objectives
-------------------

By the end of this course, you will be able to:

- **Diagnose** site effects and batch artifacts in multi-site neuroimaging datasets
- **Implement** location-scale harmonization methods (ComBat, ComBat-GAM, CovBat)
- **Evaluate** harmonization quality using appropriate metrics and validation strategies
- **Handle** longitudinal and test-retest data with specialized techniques (Long-Combat, BART)
- **Integrate** harmonization into machine learning pipelines and statistical analyses
- **Apply** advanced alternatives including deep learning and normative modeling approaches

Course Structure
-------------------
The course is divided into four main blocks, each focusing on a specific aspect of harmonization:  

- **Block 1: Effects of Site Introduction**  
  - B1_N1_Intro_To_Eos
  - B1_N2_Simulated_example_of_Eos
- **Block 2: Harmonization Evaluation**
  - B2_N1_HarmonisationEvaluation
  - B2_N2_SiteRegression_forHarmonisation
  - B2_N3_Regression_versus_ComBat_harmonisation
- **Block 3: Location-Scale Methods: ComBat Applications**
  - B3_N0_ComBat_applications
  - B3_N1_ComBat_limitations
  - B3_N2_EoS_in_ML
  - B3_N3_Metrics_by_Site
  - B3_N4_ComBat_in_imbalance_classes  
- **Block 4: Alternatives & Future Directions**
  - B4_N1_IQM-harmonisation_light
  - B4_N2_HarmoniseImages
  - B4_N3_normative_modelling

.. toctree::
   :maxdepth: 1
   :caption: Start to Finish Harmonisation material

   course_material/B1_N1_Intro_To_Eos
   course_material/B1_N2_Simulated_example_of_Eos
   course_material/B2_N1_HarmonisationEvaluation
   course_material/B2_N2_SiteRegression_forHarmonisation
   course_material/B2_N3_Regression_versus_ComBat_harmonisation
   course_material/B3_N0_ComBat_applications
   course_material/B3_N1_ComBat_limitations
   course_material/B3_N2_EoS_in_ML
   course_material/B3_N3_Metrics_by_Site
   course_material/B3_N4_ComBat_in_imbalance_classes
   course_material/B4_N1_IQM-harmonisation_light
   course_material/B4_N2_HarmoniseImages
   course_material/B4_N3_normative_modelling
   

Acknowledgement
-------------------

This course was developed as one of the outcomes of OxCIN’s (Oxford University Centre for Integrative Neuroimaging) collaborative work through the Harmonisation Working Group. It was designed by members of the group, including both internal and external collaborators, and was presented at OHBM 2026 annual meeting. We are grateful to the other members of the Harmonisation Working Group for their regular contributions to discussions and for helping shape the broader collaborative context of this work. Additionally, this work was supported by the Rosetrees Trust and the NIHR Oxford Health Biomedical Research Centre (NIHR203316). The views expressed are those of the author(s) and not necessarily those of the NIHR or the Department of Health and Social Care. The Centre for Integrative Neuroimaging was supported by core funding from the Wellcome Trust (203139/Z/16/Z and 203139/A/16/Z). 