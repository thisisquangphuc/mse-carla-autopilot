import cv2
import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET

input_dir = 'dataset/calar-front-cam'
# input_dir = 'dataset/out_'
xml_dir = input_dir
sem_dir = 'dataset/semantic'
out_ped = 'dataset/traffic_light'
out_no = 'dataset/no_traffic_light'
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_name}: {e}")
        
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created: {path}")
    else:
        print(f"Exists: {path}")
        # delete all files in path
        clear_folder(path)
        
ensure_dir(sem_dir)
ensure_dir(out_ped)
ensure_dir(out_no)
 
def classifyByXml():
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        try :
            
            xml_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            filename = root.find("filename").text
            image_path = os.path.join(input_dir, filename)

            found_pedestrian = False
            for obj in root.findall("object"):
                label = obj.find("name").text.lower()
                if label == "pedestrian":
                    found_pedestrian = True
                    break

            if found_pedestrian:
                shutil.copy(image_path, os.path.join(out_ped, filename))
            else:
                shutil.copy(image_path, os.path.join(out_no, filename))  
        
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
          
         
def classifyByOpenCV():
     
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(image_extensions):
            img_path = os.path.join(input_dir, fname)
            
            name, _ = os.path.splitext(fname)
            sem_path = os.path.join(input_dir, name + '.png')  # semantic mask from semantic folder
            # sem_path2 = os.path.join(input_dir, name + '.png')
            if not os.path.exists(sem_path):
                print(f"⚠️ Semantic file not found for: {sem_path}")
                continue
            
            sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
            # sem_img2 = cv2.imread(sem_path2, cv2.IMREAD_GRAYSCALE)
            if sem_img is None:
                print(f"❌ Failed to read semantic image: {sem_path}")
                continue

            # 4 = pedestrian class; threshold for number of pedestrian pixels
            pedestrian_mask = (sem_img == 4)
            # pedestrian_mask2 = (sem_img2 == 4)
            if np.sum(pedestrian_mask) > 100:
                shutil.copy(img_path, os.path.join(out_ped, fname))
            else:
                shutil.copy(img_path, os.path.join(out_no, fname))
            # if np.sum(pedestrian_mask2) > 100:
            #     shutil.copy(img_path, os.path.join(out_ped, fname))
            # else:
            #     shutil.copy(img_path, os.path.join(out_no, fname))
        else:
            continue
        
#####
classifyByXml()