#
from sklearn.decomposition import PCA
import os
import cv2
import time
import onnxruntime
import numpy as np
from collections import Counter
import imgaug.augmenters as iaa
from sklearn.metrics.pairwise import cosine_similarity

CL_MEAN = np.asarray([[0.485, 0.456, 0.406]], dtype=np.float32)
CL_STD = np.asarray([[0.229, 0.224, 0.225]], dtype=np.float32)


class Classifier:
    """
    Class for classification.
    Embedding model is used to generate embeddings for images. And compare it with base.
    """
    def __init__(self, model_path, image_width, image_height, 
                 base_embs_path, base_names_path, n_samples=3, sim_threshold=0.5):
        """

        :param model_path: string, path to embeder model weights <br>
        :param image_width: int, image width for embeder model <br>
        :param image_height: int, height width for embeder model <br>
        :param base_embs_path: string, path to emb base files <br>
        :param base_names_path: string, path to emb base files <br>
        :param n_samples: int, number of nearest samples to chose class <br>
        :param sim_threshold: float, min confidence of similarity <br>
        """
        
        self.model_path = model_path
        self.base_embs_path = base_embs_path
        self.base_names_path = base_names_path
        self.image_width = image_width  
        self.image_height = image_height 
        self.sim_threshold = sim_threshold
        self.n_samples = n_samples
        
        self.base_embeddings = None
        self.base_names = None
       
        try:
            self.initialize_base()
            # load model
            self.model = onnxruntime.InferenceSession(self.model_path,
                                                      providers=['CPUExecutionProvider'])
            """embedding model"""
            self.model.disable_fallback()
            self.model_input_name = self.model.get_inputs()[0].name
            """embedding model input names"""
            self.model_output_name = self.model.get_outputs()[0].name
            """embedding model output names"""
            # warmup model
            self.apply_emb_model(np.ones((image_height, image_width, 3),
                                         dtype=np.uint8))
       
        except Exception as exc:
            self.model = None
            print(f'Error product cl init: {exc}')


    def initialize_base(self):
        """
        Load base names and embeddings and initialize them<br>
        """
        # if exist load
        if os.path.exists(self.base_embs_path) and os.path.exists(self.base_names_path):
            with open(self.base_embs_path, 'rb') as embeddings_file:
                embeddings = np.load(embeddings_file)
            print(f'download base embeddings')
            with open(self.base_names_path, 'r') as names_file:
                names = np.asarray([i.rstrip() for i in names_file.readlines()])
            print(f'download base names')
        else:
            embeddings = np.asarray([])
            names = np.asarray([])
            print(f'No embeddings or names files, embeddings not loaded')

        self.base_embeddings = embeddings
        self.base_names = names


    def preprocess(self, image):
        """
        Preprocess image for model<br>
        """
        img_in = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)  # resize
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)  # convert to RGB

        img_in = img_in.astype(np.float32) / 255.0
        img_in = (img_in - CL_MEAN) / CL_STD
        img_in = np.transpose(img_in, (2, 0, 1))
        # img_in = img_in.astype(np.float16)
        return img_in
        
    
    def augment_image(self, image):
        """
        Augment image for dataset creation<br>
        """
        aug_list = [[iaa.GaussianBlur(3), iaa.Sharpen(1), iaa.SaltAndPepper(0.05)],  # Sharpen or blur
                    #[iaa.Multiply(1.5), iaa.Multiply(0.5)],  # Make brighter or darker.
                    [iaa.Affine(rotate=90), iaa.Affine(rotate=-90), iaa.Affine(rotate=180)]]  # Rotate
        images_aug = [image]
        for augs in aug_list:
            new_images = np.zeros_like([image])
            for aug in augs:
                aug_image = aug.augment_images(images_aug)
                new_images = np.concatenate([new_images, aug_image])
            images_aug += [i for i in new_images[1:]]
        return images_aug

    
    def apply_emb_model(self, image):
        """
        Apply model to image or images<br>
        """
        model_input = []
        if (not isinstance(image, list) or 
           (isinstance(image, np.ndarray) and len(image.shape) == 3)):
            image = [image]
        for img in image:
            img_in = self.preprocess(img)
            img_in = np.ascontiguousarray(img_in)
            model_input.append(img_in)

        onnx_input_image = {self.model_input_name: model_input}
        output, = self.model.run(None, onnx_input_image)
        return output
        
     
    def classify(self, images):
        """
        Classify image or list of images<br>
        """
        if self.base_embeddings is None or self.model is None or self.base_names is None:
            return None
        
        if (not isinstance(images, list) or 
           (isinstance(images, np.ndarray) and len(images.shape) == 3)):
            images = [images]
            
        query_emb = self.apply_emb_model(images)
        sims = cosine_similarity(query_emb, self.base_embeddings)
        sort_ids = np.argsort(sims, axis=1)
        
        predicted_class = []
        for i, ids in enumerate(sort_ids[:, -self.n_samples:]):
            unique, pos = np.unique(self.base_names[ids], return_inverse=True) 
            counts = np.bincount(pos)                    
            maxpos = [k for k, c in enumerate(counts) if c == max(counts)]
            unique_sim = [] 
            for k in maxpos:
                unique_sim.append(np.mean([sims[i, ids[p_i]] for p_i, p in enumerate(pos) if p == k]))
            predicted_class.append(unique[maxpos[np.argmax(unique_sim)]])
            
        return predicted_class if len(predicted_class) > 1 else predicted_class[0]



if __name__ == '__main__':
    cl = Classifier('paddle_model.onnx', 224, 224,
                    'base_embeddings.npy', 'base_names.txt')

    s_t = time.time()
    img = cv2.imread('../flickr_logos_27_dataset_images/2962045.jpg')
    pred = cl.classify(img)
    print(pred)    
    print(time.time() - s_t)
