#### EĞİTİLMİŞ YOLOV3 ile resimden nesne tespiti  ###
"""
Algoritma 
1- resim ve en boy bilgileri okunur
2-resim modele sokulmak için blob formatına çevrilir
3- kutucuk renkleri için random liste oluşturulur
4-modelin katmanları ve çıktı katmanı alınır(output alınabilmesi için)
5--model içe aktarılır ve resim girdi olarak verilir
6- çıktı katmanındaki çıktılara göre kutucuk çizdirme ve skor yazdırma gibi işlemler yapılur
"""
"""
!!!!!NOT: bu kod dosyası bazı eksikliklere sahiptir.

"""
import cv2 
import numpy as np 
from random import randint

#resim yükleme
img=cv2.imread("img\\laptop.jpg")

# resmin en ve boy bilgileri alınır
img_width = img.shape[1]
img_height = img.shape[0]

# resim blob formata çevrilecek 
# blob resmin 4 boyutlu tensörlere çevrilmiş halidir

img_blob= cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)#1/255 stabil parametre
#swapRB-RGB ye dönüştürme lçöççöç

# hazır model yaklaşık 80 adet nesne tanıyabilir
# etiketler içerisindeki isimler o nesneler
labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
            "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
            "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
            "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
            "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
            "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
            "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
            "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
            "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

# her nesnenin kutusunun farklı renkte olması için :

# 80 satır, her satırda 3 elemanlık bir dizi olacak şekilde boş bir NumPy dizisi oluşturuyoruz
colors = np.empty((80, 3), dtype=int)  
for i in range(80):
    # Rastgele bir renk oluşturuyoruz
    color = (randint(0, 255), randint(0, 255), randint(0, 255))  
    # Oluşturduğumuz rastgele rengi dizimize ekliyoruz
    colors[i] = color  


#model koda dahil edilir
model = cv2.dnn.readNetFromDarknet("model\\yolov3.cfg","model\\yolov3.weights")
# katmanlar çekilir
layers = model.getLayerNames()

#çıktı katmanları alınır
output_layer= [layers[layer-1] for layer  in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

#çıktı katmanlarının içerisindeki değerleri çıkardık
detection_layers= model.forward(output_layer)

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        #güven skoruna erişildi
        scores=object_detection[5:]
        # maksimum tahmin skorunun bulunduğu index alınır
        predicted_id =np.argmax(scores)
        # skorlar kaydedilir
        confidence=scores[predicted_id]     

        if confidence >0.60:
            # tespit edile
            label = labels[predicted_id]
            #tespitin kordinatları resmin boyutları ile oranlanarak çekilir
            bounding_box=object_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
            
            #tespit katmanından çekilen tespit kutusu bilgileri noktalar ile çekilir
            (box_center_x,box_center_y,box_width,box_height) = bounding_box.astype("int")
            
            # merkez konumundan enin yarısı çıkarılırsa kutunun başlangıç x kordinatı bulunur
            start_x = int(box_center_x-(box_width/2))
            # merkez konumdan yükseklik çıkarılınca da başlangıç y kordinatı ortaya çıkar
            start_y = int (box_center_y-(box_height/2))

            # son x kordinatı başlangıca en eklenerek bulunur
            end_x = start_x+box_width
            # son y kordinatına başlangıca yükseklik eklenerek bulunur
            end_y = start_y + box_height

            # kutu rengi seçildi
            box_color=colors[predicted_id]
            
            # kutu rengi liste halinde değişkene atılır
            box_color=[int(each) for each in box_color]


            label="{}: {:.2f}%".format(label,confidence*100)
            print("predicted object {}".format(label))


            # kordinatlara göre ve belirlenen renge göre kutu çizilir
            cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,1)

            cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)

cv2.imshow("detection",img)
cv2.waitKey(0)



            
           