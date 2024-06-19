Image retreaval system. 
 
Dataset: [Flickr Logos 27 dataset](http://image.ntua.gr/iva/datasets/flickr_logos/) (To get model use paddle_model_convert.ipynb notebook) <br>
Embedder: [Paddle Image REcognition](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/quick_start/quick_start_recognition_en.md)


To classify image used <b>3</b> nearest image from base by cosine similarity metric. For current dataset F1 score = 0.6 <br>
![TPR](https://github.com/Zhovtukhin/logo-retrieval/blob/main/TPR.png)

Use ```product_classifier.py``` to classify image. Class classifier requeires model_path, image_width, image_height, base_embs_path, base_names_path parameters. <br>
- Method ```initialize_base``` read base from file and save it in memory;
- ```preprocess``` prepare numpy imagge for model;
- ```augment_image``` create multiple images from 1 to create dataset with different variations of image (blue, rotations, etc);
- ```apply_emb_model``` run model (also call preprocess funtion);
- ```classify``` apply model and return name of class.


Most wrong results may couse by:
- Low similarity score (most correct predictions above 0.5)
- Similar logo color or texture
- Similar logo shape
- Base embeddings was created from croped logos but query images create from all image. So was comared all queris image with base not just logo
- Simple embedder

Possible improvement:
- Better model
- Different augmantation
- Combine with other fetaure extractor ( for excample based on descriptors)
- Detect logo from image before classification
