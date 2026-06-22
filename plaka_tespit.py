import cv2
import numpy as np
import easyocr
import re
import os
import matplotlib.pyplot as plt

def adimlari_ekranda_goster(gorseller_sozlugu):
    """
    Sözlük formatında verilen ara aşama görüntülerini 
    Matplotlib ile tek bir pencerede alt ızgaralar halinde gösterir.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten() 

    for i, (baslik, gorsel) in enumerate(gorseller_sozlugu.items()):
        if i >= len(axes):
            break 
            
        if len(gorsel.shape) == 3: 
            gorsel_gosterim = cv2.cvtColor(gorsel, cv2.COLOR_BGR2RGB)
            axes[i].imshow(gorsel_gosterim)
        else: 
            axes[i].imshow(gorsel, cmap='gray')
            
        axes[i].set_title(baslik, fontsize=14, fontweight='bold')
        axes[i].axis('off') 

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout() 
    plt.show()

def plaka_formatina_uygun_mu(metin):
    """
    Metni temizler, uzunluk, karakter ve 81 il kodu sınırına göre doğrular.
    """
    temiz_metin = re.sub(r'[^A-Z0-9]', '', metin.upper())
    
    if 5 <= len(temiz_metin) <= 8:
        if any(c.isalpha() for c in temiz_metin) and any(c.isdigit() for c in temiz_metin):
            if re.match(r'^\d{2}[A-Z]+\d+', temiz_metin):
                
                il_kodu = int(temiz_metin[:2])
                if 1 <= il_kodu <= 81:
                    return True, temiz_metin
                else:
                    print(f"Uyarı: İl kodu ({il_kodu}) geçerli aralıkta değil (01-81).")
            
    return False, temiz_metin

def plaka_tespit_et_ve_gorsellestir(resim_yolu):
    if not os.path.exists(resim_yolu):
        print(f"Hata: Görüntü dosyası bulunamadı: {resim_yolu}")
        return

    print("Görüntü okunuyor...")
    img = cv2.imread(resim_yolu)
    if img is None:
        print("Hata: Görüntü okunamadı.")
        return

    gorsel_adimlari = {}
    gorsel_adimlari["1. Orijinal Goruntu"] = img.copy()

    print("OCR Modeli yükleniyor, lütfen bekleyiniz...")
    try:
        reader = easyocr.Reader(['tr', 'en'], gpu=False, verbose=False)
    except Exception as e:
        print(f"Hata: OCR modeli yüklenemedi. {e}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gorsel_adimlari["2. Gri Tonlama (Grayscale)"] = gray
    
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
    gorsel_adimlari["3. Gurultu Azaltma (Bilateral)"] = bfilter
    
    blurred = cv2.GaussianBlur(bfilter, (5, 5), 0)
    
    edged = cv2.Canny(blurred, 30, 200) 
    gorsel_adimlari["4. Kenar Tespiti (Canny Edge)"] = edged
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    
    best_candidate = None
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            oran = float(w) / h
            
            if 1.5 <= oran <= 7.0:
                best_candidate = approx
                break 

    if best_candidate is not None:
        x_final, y_final, w_final, h_final = cv2.boundingRect(best_candidate)
        
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [best_candidate], 0, 255, -1)
        
        (ys, xs) = np.where(mask == 255)
        (y1, x1) = (np.min(ys), np.min(xs))
        (y2, x2) = (np.max(ys), np.max(xs))
        kirpilmis_plaka_gray = gray[y1:y2+1, x1:x2+1]
        
        gorsel_adimlari["5. Kirpilan Aday Bolge"] = kirpilmis_plaka_gray
        
        ocr_results = reader.readtext(kirpilmis_plaka_gray, detail=0)
        
        plaka_metni = ""
        is_valid_format = False
        
        if ocr_results:
            temizlenmis_metin_parcalari = []
            for text in ocr_results:
                if text.upper() != "TR":
                    temizlenmis_metin_parcalari.append(text.upper())
            
            tum_metin = "".join(temizlenmis_metin_parcalari)
            is_valid_format, temiz_plaka = plaka_formatina_uygun_mu(tum_metin)
            
            if is_valid_format:
                plaka_metni = temiz_plaka
                print(f"Plaka Tespit Edildi! Okunan Değer: {temiz_plaka}")
            else:
                plaka_metni = tum_metin
                print(f"Plaka Formatı Hatalı veya Okunamadı. Okunan Değer: '{tum_metin}'")
        else:
            plaka_metni = "Yazi Bulunamadi"
            print("Plaka bölgesinde metin algılanamadı.")

        if is_valid_format:
            color = (0, 255, 0)
            status_text = plaka_metni
        else:
            color = (0, 0, 255)
            if plaka_metni != "Yazi Bulunamadi":
                status_text = f"Format Hatali: {plaka_metni}"
            else:
                status_text = "Plaka Degil (Yazi Yok)"
            
        cv2.rectangle(img, (x_final, y_final), (x_final + w_final, y_final + h_final), color, 3)
        cv2.putText(img, status_text, (x_final, y_final - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        
        gorsel_adimlari["6. Final Tespit Sonucu"] = img
        adimlari_ekranda_goster(gorsel_adimlari)

    else:
        print("Görüntüde plaka formatında bir dikdörtgen nesne bulunamadı.")
        cv2.putText(img, "Plaka Bulunamadi", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
        
        gorsel_adimlari["5. Final Tespit Sonucu"] = img
        adimlari_ekranda_goster(gorsel_adimlari)

if __name__ == "__main__":
    test_resmi = r"C:\Users\Msi\Desktop\b.png" 
    plaka_tespit_et_ve_gorsellestir(test_resmi)