import cv2
def numara_fete(imagine_path):
    # Incarca imaginea folosind OpenCV, functia cv2.imread(..) returneaza matricea care reprezinta imaginea
    # Fiecare element al matricei corespunde valorii unui pixel din imagine
    imagine = cv2.imread(imagine_path)

    # Verifica daca imaginea a fost incarcata corect
    if imagine is None:
        print(f'Eroare la incarcarea imaginii de la calea: {calea_imaginii}')
        return

    # Converteste imaginea din spatiul de culoare (Blue-Green-Red) la scala de gri (pentru eficienta)
    imagine_gri = cv2.cvtColor(imagine, cv2.COLOR_BGR2GRAY)

    # Incarca clasificatorul preantrenat pentru recunoasterea fetei
    clasificator_fata = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Verifica daca clasificatorul a fost incarcat corect
    if clasificator_fata.empty():
        print('Eroare la incarcarea clasificatorului pentru recunoasterea fetei.')
        return

    # Detecteaza fetele in imagine
    fete = clasificator_fata.detectMultiScale(imagine_gri, scaleFactor=1.1, minNeighbors=5)

    # Numara fetele
    numar_fete = len(fete)

    # Afiseaza rezultatul
    print(f'Numarul de fete detectate: {numar_fete}')

    # Deseneaza un dreptunghi in jurul fiecarei fete detectate
    for (x, y, w, h) in fete:
        cv2.rectangle(imagine, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Afiseaza imaginea cu dreptunghiurile desenate
    cv2.imshow('Fete detectate', imagine)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Exemplu
calea_imaginii = ('pozacatell.jpg')
numara_fete(calea_imaginii)