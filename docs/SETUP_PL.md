# Instrukcja dla zespołu: instalacja i zbieranie danych

Czas: około 30 min instalacja + 40-60 min nagrywanie cyfr.

## 1. Co trzeba mieć

- **Python 3.11** (nie 3.12, 3.13 ani 3.14: MediaPipe 0.10.21 nie ma dla nich paczek).
  Windows: instalator z https://www.python.org/downloads/release/python-3119/
  (zaznacz „py launcher”). Sprawdź: `py -3.11 --version`.
- **Git**: https://git-scm.com/download/win (Windows) albo `brew install git` / `apt install git`.
- **Konto GitHub** z dostępem do repozytorium (Mateusz doda Cię jako współpracownika,
  zaproszenie przyjdzie mailem, trzeba je zaakceptować).
- **Kamerka** (wbudowana wystarczy) i w miarę równe światło, ręka na tle ściany.

## 2. Pobranie projektu i środowisko

```bash
git clone https://github.com/MateuszDobry/airwrite.git
cd airwrite
py -3.11 -m venv .venv          # macOS/Linux: python3.11 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Jeśli `pip` się zawiesza przy pobieraniu torcha, dodaj
`--timeout 300 --retries 10` (albo mirror `-i https://pypi.tuna.tsinghua.edu.cn/simple`).

Każde polecenie poniżej uruchamiaj **z aktywnym `.venv`** (na początku linii widać `(.venv)`)
i zawsze z `python` na początku (samo `whiteboard.py` otwiera plik w edytorze).

## 3. Test, czy wszystko działa

```bash
python gestures.py                       # bez kamery, kilka sekund
python train.py --quick                  # ok. 1-2 min, pierwszy raz pobiera MNIST
python ablation.py --quick --synthetic
python analysis.py --quick --synthetic
python whiteboard.py                     # kamera: rysuj palcem wskazującym
```

Gesty: **palec wskazujący** rysuje, **pięść** pauza, **wskazujący + środkowy**
przesuwa kursor bez rysowania, **otwarta dłoń** czyści. `q` zamyka okno.
Jeśli otwiera się zła kamera, zmień `CAMERA_INDEX` w `config.py` (tylko lokalnie, nie commituj).

Wyniki z `--quick`/`--synthetic` trafiają do folderów `smoke/` i `synthetic/`: to nie są
prawdziwe wyniki i nie wolno ich cytować w raporcie.

## 4. Zbieranie danych

Imię wpisz **małymi literami, bez polskich znaków i spacji** (np. `kasia`, `jan`):

```bash
python collect_data.py --writer <imie> --per-digit 20
```

Skrypt prowadzi przez cyfry 0-9, po 20 próbek każdej (razem 200 obrazków).

| Klawisz | Działanie |
|---|---|
| SPACJA | zapisz i przejdź do kolejnej próbki |
| `c` | wyczyść bez zapisywania |
| `n` / `b` | następna / poprzednia cyfra |
| `q` | wyjdź (postęp zostaje, można uruchomić ponownie i kontynuować) |

Zasady (ważne dla wyników):

- **Pisz naturalnie, swoim tempem.** Nie próbuj, żeby cyfry wyglądały jak z MNIST:
  właśnie tę różnicę mierzymy.
- Nie zmieniaj `STROKE_THICKNESS` ani `SMOOTHING_WINDOW` w `config.py`: wszyscy muszą
  nagrywać z tymi samymi ustawieniami.
- Na koniec zajrzyj do `data/raw/<imie>/`: każdy folder 0-9 powinien mieć 20 plików.
  Usuń tylko ewidentnie zepsute (pusty lub urwany rysunek) i zapisz, ile usunąłeś.
  Brakujące dograj, uruchamiając skrypt jeszcze raz.

## 5. Oddanie danych (Pull Request)

```bash
git checkout -b data-<imie>
git add data/raw/<imie>/
git commit -m "Add digits from <imie>"
git push -u origin data-<imie>
```

Potem na GitHubie kliknij **Compare & pull request** i utwórz PR. Mateusz go zmerguje.
Dodawaj **tylko** swój folder `data/raw/<imie>/`, bez zmian w kodzie i w `config.py`.

Jeśli Git sprawia problemy: spakuj folder `data/raw/<imie>/` do zipa i wyślij Mateuszowi.

## 6. Co dalej

Gdy dane wszystkich są w repo, osoba B uruchamia `airwrite_colab.ipynb` na Colabie
(trening + eksperymenty + wykresy).
