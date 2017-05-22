Dokumentcia - zložka so zdrojovými kódmi dokumentácií k bakalárskej práci a pdf súbory bakalárskej práce

Video - zložka s anglickou a slovenskou verziou prezentačného videa

ZdrojoveKody - zložka so skriptami a natrénovanými modelmi
	-> anotacie - ukážka anotácií videí v dátovej sade
	-> LinearneRegresory
		-> get(Image|Sound)Activation.py - script na získanie príznakov zo zvuku/videa
		-> (image|sound)Activation.sh - bash skript, ktorý spúšta získavanie postupne po súboroch
		-> Image - zložka s natrénovanými lineárnymi regresormi pre obrázky
		-> Sound - zložka s natrénovanými lineárnymi regresormi pre zvuk
	-> Bash - zložka s bash-ovskými skriptami na extrakciu snímkov a zvuku z videa a konverziu zvuku
	-> CNN - zložka so skriptami a modelmi ku konvolučným neurónovým sieťam
		-> spectrograms.py - skript na vykreslenie spektrogramu z .wav súboru
		-> createSpectrograms.sh - skript na postupné vykreslenie spektrogramov zo všetkých súborov
		-> test_net_(regre|classification).py - skritpy, ktoré testujú úspešnosť jednotlivých modelov
		-> test_models_(regre|classification).sh - skripty, ktoré púšťajú vyhodnocovanie všetkých modelov
		-> train_(regre|classification).py - skripty použité na trénovanie sietí
		-> RegresiaZvuku - zložka s experimentami s regresiou nad spektrogramami
			-> MalaSiet - experiment so sietov 2 konv. vrstvy + 1 plne prepojená
				-> deploy.prototxt - architektúra siete pripravená na použitie
				-> net_solver.prototxt - odpoveďový súbor pre trénovanie siete
				-> net_train_val.prototxt - architektúra sieťe použitá na trénovanie
				-> trained.caffemodel - natrénovaný model siete
				-> vysledky.txt - vyhodnotený súbor s výsledkami modelu
			-> VelkaSiet - experiment so sietov 6 konv. vrstiev a 3 plne prepojené vrstvy
		-> KlasifikaciaZvuku - zložka s experimentami s klasifikačným riešením
			-> get_bins.py - skript na zisťovanie priemerných hodnôt pre jednotlivé triedy
			-> 3-vrstvovaSiet - experiment s 3 vrstvovou kovolučnou neurónovou sieťou
			-> 4-vrstvovaSiet - experiment s 4 vrstvovou kovolučnou neurónovou sieťou
		-> Obraz - zložka s experimentami s obrazovou modalitou
			-> featurePreview.txt - ukážka príznakov z nástroja OpenFace
			-> process.py - skript na spracovanie príznakov z OpenFace
			-> Pohlad - experiment s konvolučnou neurónovou sieťou spracovávajúcou pohľad
			-> OrientacneBody - experiment s konvolučnou neurónovou sieťou spracovávajúcou orientačné body tváre
				-> sizeNorm.caffemodel - model s použitím normalizácie podľa veľkosti detekovaného okna
				-> moveNorm.caffemodel - model s použitím normalizácie podľa pohybu orientačných bodov tváre  
