
import tensorflow as tf

def review_data_spec(
        dataset : tf.data.Dataset, reviews : int = 1, plot_img : bool=True
):
    import matplotlib.pyplot as plt
    for features in dataset.take(reviews):
        print(f"\n\n -------- id : {features['meta_info']['id']}-----------")
        for key in features.keys():
            if isinstance(features[key], tf.Tensor):
                print(f"{key}: shape - {features[key].shape}")
            else:
                assert isinstance(features[key], dict)
                print(f"{key}: ")
                for sub_key in features[key].keys():
                    #print(f"{sub_key}: sub_dict - [{features[key][sub_key]}]") 
                    shape = features[key][sub_key].shape
                    print(f"      {sub_key}: {shape if shape != () else features[key][sub_key]}")

        # if plot_img :       
        # from lib.visualization import Vis_SampleTransform
        # if plot_img :
        #     plt.figure(figsize=(8, 8))
        #     text  =   f"instance_id : {features['meta_info']['id']} \n"
        #     text  += f"(image_id: {features['meta_info']['image_id']} )"
        #     plt.title(text, fontsize = 12)
        #     plt.imshow(features["image"].numpy())

        #     if features.get('kps',None) is not None :
        #         total_kps = features['kps']
        #         plt.scatter(*zip( *[(kps[0], kps[1])for kps in total_kps ]),s=10)
         
        #     if features.get('bbox',None) is not None :
        #         ax = plt.gca()  
        #         x1, y1, w, h = features['bbox']
        #         patch = plt.Rectangle(
        #                 [x1, y1], w, h, 
        #                 fill=False, 
        #                 edgecolor=[1, 0, 0], 
        #                 linewidth=2
        #         )
        #         ax.add_patch(patch)

