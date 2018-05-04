from OCR import *
from extract_characters import *
from pathlib import Path
import pickle
from PIL import Image, ImageDraw
import cv2

def classify(model,pp,images):
    feature_list = []
    for i in range(len(images)):
        image = images[i]

        df = hog_feature_extraction(image)
        df = pp.transform(np.array([df],'float64'))

        feature_list.append(df)

    results = []
    for i in range(len(feature_list)):
        #Prediction
        predict = model.predict(feature_list[i].reshape((1,-1)))[0]
        prob = model.predict_proba(feature_list[i].reshape((1,-1)))
        results.append([images[i],predict,prob.max()])

    return results

def process_classifications(im,clas_segments,cord,results,prob_threshold,name,save):
    output_dir = "output/classifications/"
    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    im = np.asarray(Image.fromarray(im).convert("RGB"))
    classified_images = []

    for t in range(len(results)):
        image_prob = results[t][2]
        if(image_prob>=prob_threshold): 
            image_class = label_to_letter(int(results[t][1]))
            classified_images.append(t)
            #print("Image nr {}: labeled: {}, prob: {:0.5f}".format(t,image_class,image_prob))

            #add bounding box to image
            pt1,pt2 = (cord[t][0],cord[t][1]),(cord[t][0]+22,cord[t][1]+22)
            im = cv2.rectangle(im,pt1,pt2,(0,255,0),2)

            if(save):
                image_name = "{}_{}_{}_{:0.2f}".format(name,t,image_class,image_prob)
                seg_image = Image.fromarray(clas_segments[t]).convert('RGB')
                seg_image.save(output_dir+image_name+".jpg")
 
    im = Image.fromarray(im)
    im.save(output_dir+name+"_BB.jpg")


def main():
    #----------Training-------------
    load = False #Load earlier models
    feature_method = "HOG"
    classification_method = "SVM"

    col_dir = 'chars74k-lite/*/*.jpg'
    print("===Training: ",col_dir)

    #Create data sets from training iterations
    label,image = get_image(col_dir)
    train_images,test_images,train_labels,test_labels = split_train_test(label,image)
    
    #Create model
    if(load and Path("saved_model.o").is_file() 
            and Path("saved_pp.o").is_file() 
            and Path("saved_pca.o").is_file()):
        print("Loading models")
        model = pickle.load(open("saved_model.o","rb"))
        pp = pickle.load(open("saved_pp.o","rb"))
        pca = pickle.load(open("saved_pca.o","rb"))
    else:
        print("Processing training data")
        train_images = data_processing(train_images)
        print("Creating models")
        model,pp,pca = training(train_images, train_labels,feature_method,classification_method)
        print("Saving models")
        pickle.dump(model, open("saved_model.o","wb"))
        pickle.dump(pp, open("saved_pp.o","wb"))
        pickle.dump(pca, open("saved_pca.o","wb"))

   
    #-----------Testing--------------
    #Test the models on the training data
    test_model = True

    if(test_model):
        print("Testing models")
        test_images = data_processing(test_images)
        test(model,pp,pca,test_images,test_labels,feature_method,classification_method)

    #---------Classify External Images-----------
    #Classify an external image
    classify_seg = True
    if(not classify_seg): return

    clas_dir = 'detection-images/*.jpg'
    print("===Classify: ",clas_dir )

    image_number = 1    #Image number to classify in traget folder
    save_seg = False    #Save segment items

    save_name = "Im{}".format(image_number)
    clas_images = get_image_classify(clas_dir)
    clas_segments,cord = extract_windows(clas_images[image_number],save_name,save_seg)
    clas_preprocessed = data_processing(clas_segments)
    results = classify(model,pp,clas_preprocessed)

    #---------Retrieve External Classififactions-----------
    prob_threshold = 0.8
    print("===Retrieving Classifications, threshold: {}", prob_threshold)
    save_clas = True #Save classification items

    im = clas_images[image_number]
    process_classifications(im,clas_segments,cord,results,prob_threshold,save_name,save_clas)
    




if __name__ == "__main__":
    main()

