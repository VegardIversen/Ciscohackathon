import os
import cv2
import face_recognition
import imutils
from PIL import Image


face_cascade = cv2.CascadeClassifier('C:\\Users\\vegar\\Desktop\\NTNU\\Elsys\\vaar20\\ImClasif\\clas\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
KNOWN_FACES_DIR = 'known_faces'
#UNKNOWN_FACES_DIR = 'unknown_faces' trengs kun for bilder
TOLERANCE = 0.5 #kan endres
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

video = cv2.VideoCapture(0)

#funker ikke forløpig. Ønsker å rotere bildet riktig slik at den greier å finne ansiktet. Finner den ikke ansikt kommer det out of range error
def rotate_image(frame, name):
    frame = cv2.imread(f'{KNOWN_FACES_DIR}/{name}/{filename}')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces == ():
        for angle in range(0,360,15):
            print('rotating')
            rotated = imutils.rotate_bound(frame,angle)
            rotate_image(rotate_image, name)
    else:
        try:

            im = Image.open(f'{KNOWN_FACES_DIR}/{name}/{filename}')
            im.save(f'{KNOWN_FACES_DIR}/{name}/{filename}')
            print('yey')
        except IOError:
            print("cannot create thumbnail for", f'{KNOWN_FACES_DIR}/{name}/{filename}')
        


    return 
    



# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


print('Loading known faces...')
known_faces = []
known_names = []

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
       
        
       

        
        #rotate_image(filename, name)
        

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)


print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
#for filename in os.listdir(UNKNOWN_FACES_DIR): #for bilde

while True:

    # Load image
    #print(f'Filename {filename}', end='') #for bilder
    #image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}') #for bilder

    ret, image = video.read()
    


    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)

    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)

    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #for bilder

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')

            # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = name_to_color(match)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
    cv2.imshow(filename, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    # Show image
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL) #for bilder
    #im = cv2.imread(image)
    #imS = cv2.resize(image,(960,540)) #for bilder
    # #cv2.resizeWindow('image', 600,600)
    #cv2.imshow('image', imS) #for bilder
    #cv2.waitKey(0) #for bilder
    #cv2.destroyWindow(filename) #for bilder
    #cv2.imshow(filename, image)
    # cv2.imshow(filename,image)
    # cv2.waitKey(0)
    #cv2.destroyWindow(image)