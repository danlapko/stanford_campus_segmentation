
## Similar datasets overview

Search on github did not give any results related to the segmentation task for stanford drone dataset;

Review of datasets mentioned here https://github.com/eric-erki/awesome-satellite-imagery-datasets,
here https://lionbridge.ai/datasets/15-best-aerial-image-datasets-for-machine-learning/ and in google search for transfer learning purposes:

* dd-ml-segmentation-benchmark. Here only 'car' class overlaps with our dataset, also most of the cars are parked and have very small size, seems not very usefull for transfer learning
https://github.com/dronedeploy/dd-ml-segmentation-benchmark

* DLR-SkyScapes: Aerial Semantic Segmentation Dataset for HD-mapping.
Here there are some vehicle classes (car, trailer, van, truck, large truck, bus), images have similar zoom as stanford dataset.
In total there are 16 images of size 5616x3744, dataset could be accessed only by request.
https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-12760/22294_read-58694

* Dstl Satellite Imagery Feature Detection. Include small and large vehicle classes. Good article about competition and training process.
https://habr.com/ru/company/avito/blog/325632/

* ISPRS Test Project on Urban Classification and 3D Building Reconstruction. Contains car objects. Data available upon request.
https://www2.isprs.org/commissions/comm2/wg4/benchmark/semantic-labeling/

* [Good] Aerial Semantic Segmentation Drone Dataset. 400 images at a size of 6000x4000px (24Mpx).
Contains following classes: person, dog, car, bicycle. https://www.kaggle.com/bulentsiyah/semantic-drone-dataset

* [Good] UAVID 2020 segmentation dataset. Contains 30 sequence of  4k frames labeled in 4 including following categories: Moving car, Static car, Human.
https://uavid.nl/ . Camera orientation is not vertical, tilt angle ~45 grad

* [Good] Aeroscapes. Dataset comprises of images captured using a commercial drone from an altitude range of 5 to 50 metres. The dataset provides 3269 720p images and ground-truth masks for 11 classes including Car, Bike, Person, Animal.
Camera orientation varies from vertical to horizontal.
https://github.com/ishann/aeroscapes