Справочник разработчика ИТС: 
Модули видеоаналитики и обработки данныхДанный ресурс предназначен для изучения принципов построения систем мониторинга дорожного движения с использованием компьютерного зрения и распределенных хранилищ.1. Инициализация системных интерфейсовПри разработке аналитических модулей критически важно обеспечить корректное извлечение параметров конфигурации из среды окружения (Environment Variables). Это позволяет коду оставаться универсальным при переносе между серверами.1.1. Настройка сетевых соединений (Database, RTMP, S3)os.getenv — стандартный метод получения параметров. Если переменная не задана в системе, используется значение по умолчанию.Minio — инициализация клиента для объектного хранилища. Параметр secure=False используется для локальных сетей.Pythonimport os
from minio import Minio
import psycopg2

### ПРИМЕР СИНТАКСИСА ИНИЦИАЛИЗАЦИИ:
### Настройка подключения к PostgreSQL (Схема данных: userXX)
DB_URL = os.getenv("DB_URL", "host=<IP> port=<PORT> dbname=<DB> user=<USER> password=<PASS>")

### Настройка захвата RTMP-потока
RTMP_URL = os.getenv("RTMP_URL", 'rtmp://<URL>:<PORT>/stream')

### Конфигурация S3-совместимого хранилища (MinIO)
MINIO_CLIENT = Minio(
    os.getenv("MINIO_ENDPOINT", "<IP>:9000"),
    access_key=os.getenv("MINIO_KEY", "<KEY>"),
    secret_key=os.getenv("MINIO_SECRET", "<SECRET>"),
    secure=False
)
2. Геометрический анализ и пространственная детекцияЭффективность ИТС зависит от точности определения положения объектов в пространстве кадра.2.1. Трансформация координат и расчет дистанцийДля работы с перспективными искажениями на видео используются функции расчета расстояния от точки до прямой. Это позволяет фиксировать пересечение виртуальных линий контроля.Pythondef point_line_distance(px, py, x1, y1, x2, y2):
    """
    Расчет кратчайшего расстояния от центра объекта (px, py) 
    до виртуальной линии замера (x1, y1) -> (x2, y2).
    """
    line_vec = np.array([x2 - x1, y2 - y1])
    p_vec = np.array([px - x1, py - y1])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0: return np.linalg.norm(p_vec)
    # Нормализация вектора и расчет проекции точки на прямую
    t = np.clip(np.dot(line_vec / line_len, p_vec / line_len), 0, 1)
    return np.linalg.norm(p_vec - line_vec * t)
3. Трекинг и расчет скоростных характеристикИспользование нейросетевых моделей семейства YOLO с параметром persist=True позволяет сохранять уникальный идентификатор (ID) объекта на протяжении всего времени его нахождения в кадре.3.1. Алгоритм вычисления мгновенной скоростиВход в зону: Фиксация времени t1 при достижении первой линии.Выход из зоны: Фиксация времени t2 при достижении второй линии.Расчет: Скорость $V = (S / (t2 - t1)) * 3.6$, где $S$ — реальное расстояние в метрах.Python# Логика трекинга внутри итерационного цикла (пример)
results = model.track(frame, persist=True, classes=[2, 3, 5, 7]) # Автомобили и грузовики

if results[0].boxes.id is not None:
    # Извлечение координат (xyxy), идентификаторов (id) и классов (cls)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    for i, t_id in enumerate(ids):
        # Расчет центра объекта (cx, cy) для анализа пересечения линий
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
4. Регистрация событий и жизненный цикл данных4.1. Логирование в реляционные СУБДВсе данные телеметрии (координаты, скорость, тип объекта) должны записываться в таблицу фактов для последующего предиктивного анализа.Python# Структура SQL-запроса для вставки данных телеметрии
SQL_INSERT = """
INSERT INTO <SCHEMA>.full_tracking_data 
(track_id, object_type, width, length, detection_time, x_cord_m, y_cord_m, speed_km_h) 
VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s)
"""
4.2. Мониторинг безопасности (Опасные сближения)Система анализирует евклидово расстояние между активными объектами. Если дистанция в метрах (с учетом коэффициента PIXELS_PER_METER) становится меньше порога безопасности, событие фиксируется в таблице инцидентов.5. Визуализация интерфейса оператора (Overlay)Для наглядности работы алгоритмов поверх видеопотока накладывается графический слой.cv2.fillPoly — отрисовка зоны интереса (ROI).cv2.addWeighted — создание эффекта полупрозрачности для выделенных зон.
список библиотек:
import cv2  
import psycopg2  
import time  
import numpy as np  
import os  
from ultralytics import YOLO  
from datetime import datetime, timedelta  
from minio import Minio  
import io  
функции:  
def get_pixel_coords(pts, w, h):  
    return np.array([(int(p[0] * w), int(p[1] * h)) for p in pts], np.int32)  

def point_line_distance(px, py, x1, y1, x2, y2):  
    line_vec = np.array([x2 - x1, y2 - y1])  
    p_vec = np.array([px - x1, py - y1])  
    line_len = np.linalg.norm(line_vec)  
    if line_len == 0: return np.linalg.norm(p_vec)  
    line_unitvec = line_vec / line_len  
    p_vec_scaled = p_vec / line_len   
    t = np.clip(np.dot(line_unitvec, p_vec_scaled), 0, 1)  
    nearest = line_vec * t  
    return np.linalg.norm(p_vec - nearest)    
Пояснение/  
1. Подготовка векторов
Python  
line_vec = np.array([x2 - x1, y2 - y1]) # Вектор самого отрезка (линии)
p_vec = np.array([px - x1, py - y1])    # Вектор от начала линии до нашей точки
Мы представляем линию как стрелку, идущую из точки 1 в точку 2. Также мы строим стрелку от начала линии к объекту (машине).

2. Нормализация (Вычисление длины)
Python
line_len = np.linalg.norm(line_vec) # Длина отрезка в пикселях  
if line_len == 0: return np.linalg.norm(p_vec) # Если линия — это точка, просто считаем расстояние  
Здесь мы находим физическую длину линии. Если она нулевая, то расстояние до «линии» — это просто расстояние до этой точки.

3. Проекция точки на линию (Самый важный момент)
Python
line_unitvec = line_vec / line_len # Единичный вектор линии (направление)  
p_vec_scaled = p_vec / line_len    # Масштабируем вектор точки относительно длины линии  

Находим параметр t (коэффициент проекции)
t = np.clip(np.dot(line_unitvec, p_vec_scaled), 0, 1)  
Скалярное произведение (np.dot) находит «тень», которую точка отбрасывает на линию.

t — это число от 0 до 1.  

Если t=0, ближайшая к объекту точка на линии — это её начало.

Если t=1, ближайшая точка — это конец линии.

Если t=0.5, объект находится ровно напротив середины линии.

np.clip(..., 0, 1) гарантирует, что если машина уже «пролетела» мимо линии или еще не доехала до её начала (по бокам), мы будем считать расстояние именно до ближайшего кончика отрезка, а не до бесконечной прямой.

4. Поиск ближайшей точки и финальное расстояние
Python
nearest = line_vec * t # Находим координаты точки на линии, которая ближе всего к машине  
return np.linalg.norm(p_vec - nearest) # Считаем длину вектора между машиной и этой точкой  
Здесь мы вычисляем финальную длину «перпендикуляра». Это и есть искомое расстояние в пикселях.

def log_event(msg):  
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")  

def upload_snapshot(minio_client, frame):  
    """Пытается отправить в MinIO, иначе сохраняет локально."""  
    filename = f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"  
def get_s3_connection():  
    """
    Создает клиентское соединение с контейнером MinIO.
    Параметр secure=False обязателен для локальных Docker-инстансов без SSL.  
    """
    try:  
        client = Minio(  
            MINIO_ADDR,  
            access_key=ACCESS_KEY,  
            secret_key=SECRET_KEY,  
            secure=False   
        )
        # Проверка доступности бакета при инициализации
        bucket = "traffic-snapshots"  
        if not client.bucket_exists(bucket):  
            client.make_bucket(bucket)  
            print(f"📦 Бакет {bucket} успешно создан в контейнере")  
        return client  
    except Exception as e:   
        print(f"❌ Ошибка подключения к Docker-сервису MinIO: {e}")  
        return None  
отправка данных
В данной архитектуре реализована первичная фильтрация на стороне источника (Data Source Filtering). Это позволяет не перегружать канал связи и хранилище заведомо избыточными данными, которые не представляют интереса для бизнес-логики (вне зоны ROI)»
В классической аналитике больших данных (Big Data) сначала сохраняют вообще всё (даже мусор), а потом чистят. Но в твоем проекте мы используем подход IoT Edge Processing (обработка на краю).
1.	Экономия ресурсов: Если мы будем писать в бд вообще всё, что видит YOLO (кусты, небо, машины в 500 метрах), база данных переполнится за час.
2.	Техническое ограничение: Скрипт — это и есть первичный фильтр.

        try:  
            success, buffer = cv2.imencode('.jpg', frame)  
            if success:  
                file_data = io.BytesIO(buffer)  
                minio_client.put_object(  
                    BUCKET_NAME, filename, file_data, len(buffer), content_type='image/jpeg'  
                )  
                log_event(f"☁️ Снимок отправлен в MinIO: {filename}")  
                return  
Для дальнейшей работы заведем константы:
    total_count = 0  
    Будет считаться итоговым количеством обьектов
    
    entry_times, processed_ids, last_incident_time = {}, set(), {}  
    final_speeds = {}  
    last_upload_time = datetime.now() - timedelta(minutes=UPLOAD_INTERVAL_MIN)  

Управление видеопотоком и отказоустойчивость
Работа с RTMP-потоками характеризуется возможной нестабильностью сетевого соединения. В данном блоке реализован механизм автоматического переподключения: при потере кадра (not success) система делает паузу и инициирует новый захват видео.

Python
# Инициализация итерационного процесса обработки кадров
try:  
    while True:  
        # Захват кадра из сетевого буфера  
        success, frame = cap.read()  
        if not success:  
            # Механизм восстановления соединения при сбое  
            time.sleep(5)  
            cap = cv2.VideoCapture(RTMP_URL)  
            continue  

        # Определение пространственных метрик текущего кадра
        h, w = frame.shape[:2]  
        now_ts, now_dt = time.time(), datetime.now()  
        # Генерация полигональной маски зоны контроля (ROI)  
        roi_pixels = get_pixel_coords(ROI_PERCENT_POINTS, w, h)  
5.2. Детекция и интеллектуальный трекинг объектов
Для идентификации транспортных средств применяется современная нейросетевая модель, способная отслеживать перемещение объектов между кадрами. Параметр persist=True позволяет формировать временные ряды для каждого конкретного автомобиля.

Фильтрация по ROI: Система обрабатывает только те объекты, центр которых находится внутри заданной оператором зоны.

Классификация: В расчет принимаются только целевые категории транспорта (легковые, грузовые, автобусы).

Python
        # Активация инференса нейросети с фильтрацией по классам
        results = model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])  

        if results[0].boxes.id is not None:  
            # Извлечение тензоров координат, ID и классов для пакетной обработки
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(float)  
            ids = results[0].boxes.id.cpu().numpy().astype(int)  
            clss = results[0].boxes.cls.cpu().numpy().astype(int)  
            active_objects = []  

            for i in range(len(ids)):   
                # Определение геометрического центра объекта (Centroid)  
                x1, y1, x2, y2 = boxes[i]  
                t_id, cx, cy = int(ids[i]), (x1 + x2) / 2, (y1 + y2) / 2  
                
                # Проверка вхождения центроида в полигон контроля
                if cv2.pointPolygonTest(roi_pixels, (int(cx), int(cy)), False) < 0:
                    continue
5.3. Динамический расчет телеметрии и запись в DWH
Одним из ключевых показателей эффективности ИТС является сбор данных о скоростном режиме. В коде реализована логика «виртуального секундомера», замеряющего время прохождения дистанции между двумя контрольными линиями.

Данные не просто обрабатываются, но и транслируются в базу данных PostgreSQL в режиме реального времени для последующего наполнения витрин данных.

Python
                # Расчет дистанций до контрольных линий (Старт/Финиш)
                d_start = point_line_distance(cx, cy, *LINE_START_P1, *LINE_START_P2)
                d_end = point_line_distance(cx, cy, *LINE_END_P1, *LINE_END_P2)

                # Фиксация времени входа в измерительную зону
                if d_start < LINE_THRESHOLD and t_id not in entry_times:
                    entry_times[t_id] = now_ts

                # Расчет скорости при выходе из зоны
                if d_end < LINE_THRESHOLD and t_id in entry_times and t_id not in processed_ids:
                    duration = now_ts - entry_times[t_id]
                    if duration > 0.3:
                        speed = (REAL_DIST_M / duration) * 3.6
                        processed_ids.add(t_id)
                        final_speeds[t_id] = speed
5.4. Система обеспечения безопасности (Incident Management)
Алгоритм непрерывно анализирует взаимное расположение транспортных средств. При обнаружении критического сближения (дистанция менее CRITICAL_DIST_M) система генерирует предупреждение и записывает инцидент в специализированную таблицу БД для последующего разбора дорожными службами.

5.5. Визуализация и внешние интеграции
Завершающий этап цикла включает отрисовку графических интерфейсов для оператора (HUD) и периодическую выгрузку снимков ситуации на дороге в S3-хранилище (MinIO).

Python
        # Формирование многослойного изображения с наложением полупрозрачной маски ROI
        overlay = display_frame.copy()
        cv2.fillPoly(overlay, [roi_pixels], (255, 191, 0))
        cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)

        # Вывод текущих данных о трафике на экран
        cv2.putText(display_frame, f"TRAFFIC: {total_cars_count}", (25, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        # Условная выгрузка снимка в объектное хранилище по расписанию
        if (now_dt - last_upload_time).total_seconds() >= UPLOAD_INTERVAL_MIN * 60:
            upload_snapshot(minio_client, display_frame)
            last_upload_time = now_dt
finally:
cap.release()

ultralytics
opencv-python
psycopg2-binary
streamlit
pandas
scikit-learn
notebook
lapx
boto3
yt-dlp
seaborn
minio
apache-airflow
cv2.destroyAllWindows()
if db_conn: db_conn.close()



import cv2
import numpy as np
import time
import io
import os
from datetime import datetime
from ultralytics import YOLO
from minio import Minio
import psycopg2




Этот код — это ETL-процесс в реальном времени: Извлечение (Extract) данных из видео, Преобразование (Transform) координат в метры и Загрузка (Load) в базу данн


# 1. КОНФИГУРАЦИЯ И НАСТРОЙКИ (Константы)

RTMP_URL = "rtmp://your_server_ip/live/stream"  # Источник видео
PIXELS_PER_METER = 20  # Коэффициент масштабирования (подбирается по кадру)
REAL_DIST_M = 15.0     # Дистанция между контрольными линиями в метрах
LINE_THRESHOLD = 15    # Чувствительность срабатывания линии (в пикселях)

# Координаты зоны ROI (в процентах от 0.0 до 1.0 для универсальности)
ROI_PERCENT_POINTS = [(0.1, 0.4), (0.9, 0.4), (0.9, 0.9), (0.1, 0.9)]

# Линии замера скорости (x1, y1, x2, y2)
LINE_START = (200, 400, 1000, 400) 
LINE_END = (200, 700, 1000, 700)

# 2. МАТЕМАТИЧЕСКИЙ БЛОК (Логика аналитики)

def point_line_distance(px, py, x1, y1, x2, y2):
    """Вычисляет кратчайшее расстояние от точки до отрезка (Евклидова метрика)"""
    line_vec = np.array([x2 - x1, y2 - y1])
    p_vec = np.array([px - x1, py - y1])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0: return np.linalg.norm(p_vec)
    t = np.clip(np.dot(line_vec / line_len, p_vec / line_len), 0, 1)
    nearest = line_vec * t
    return np.linalg.norm(p_vec - nearest)

def get_pixel_coords(percent_points, w, h):
    """Преобразование процентов в реальные пиксели кадра"""
    return np.array([[int(x * w), int(y * h)] for x, y in percent_points], np.int32)


# 3. ИНФРАСТРУКТУРА (Подключения)


def init_connections():
    """Инициализация соединений с БД и хранилищем"""
    # MinIO (Docker)
    try:
        m_client = Minio("localhost:9000", access_key="minio_admin", secret_key="minio_password", secure=False)
        if not m_client.bucket_exists("traffic-snapshots"):
            m_client.make_bucket("traffic-snapshots")
    except: m_client = None

    # PostgreSQL
    try:
        db = psycopg2.connect(host="localhost", database="its_db", user="user43", password="password")
        db.autocommit = True
    except: db = None
    
    return m_client, db  

# 4. ОСНОВНОЙ ЦИКЛ ДЕТЕКЦИИ (ETL Процесс)


def run_detector():  
    m_client, db_conn = init_connections()  
    cursor = db_conn.cursor() if db_conn else None  
    
    model = YOLO("yolov8n.pt")  
    cap = cv2.VideoCapture(RTMP_URL)  
    
    # Словари для хранения состояний объектов  
    entry_times = {}   # {id: время_входа}  
    final_speeds = {}  # {id: скорость}  
    processed_ids = set()  
    
    try:  
        while True:  
            success, frame = cap.read()  
            if not success:  
                time.sleep(5)  
                cap = cv2.VideoCapture(RTMP_URL); continue  

            h, w = frame.shape[:2]  
            now_ts = time.time()  
            roi_pixels = get_pixel_coords(ROI_PERCENT_POINTS, w, h)  

            # --- ИНФЕРЕНС И ТРЕКИНГ ---
            results = model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])  

            if results[0].boxes.id is not None:  
                boxes = results[0].boxes.xyxy.cpu().numpy()  
                ids = results[0].boxes.id.cpu().numpy().astype(int)  
                clss = results[0].boxes.cls.cpu().numpy().astype(int)  

                for i in range(len(ids)):  
                    x1, y1, x2, y2 = boxes[i]  
                    t_id, cx, cy = ids[i], (x1 + x2) / 2, (y1 + y2) / 2  
                    
                    # 1. Фильтрация по зоне ROI (Слой ODS)  
                    if cv2.pointPolygonTest(roi_pixels, (int(cx), int(cy)), False) < 0:  
                        continue  

                    # 2. Расчет скорости (Математический блок)
                    d_start = point_line_distance(cx, cy, *LINE_START)  
                    d_end = point_line_distance(cx, cy, *LINE_END)  

                    if d_start < LINE_THRESHOLD and t_id not in entry_times:  
                        entry_times[t_id] = now_ts  

                    if d_end < LINE_THRESHOLD and t_id in entry_times and t_id not in processed_ids:  
                        duration = now_ts - entry_times[t_id]
                        if duration > 0.3:
                            speed = (REAL_DIST_M / duration) * 3.6
                            final_speeds[t_id] = speed
                            processed_ids.add(t_id)

                    # 3. Загрузка данных в DWH (Слой RAW/ODS)
                    if cursor:
                        try:
                            cursor.execute("""  
                                INSERT INTO full_tracking_data   
                                (track_id, object_type, x_cord_m, y_cord_m, speed_km_h, detection_time)   
                                VALUES (%s, %s, %s, %s, %s, NOW())  
                            """, (int(t_id), model.names[clss[i]], float(cx/PIXELS_PER_METER),   
                                  float(cy/PIXELS_PER_METER), float(final_speeds.get(t_id, 0)))  
                            )  
                        except: pass  

            # --- ВИЗУАЛИЗАЦИЯ ---
            # Отрисовка ROI и линий  
            cv2.polylines(frame, [roi_pixels], True, (255, 191, 0), 2)  
            cv2.line(frame, LINE_START[:2], LINE_START[2:], (0, 255, 0), 2)  
            cv2.line(frame, LINE_END[:2], LINE_END[2:], (0, 0, 255), 2)  
            
            cv2.imshow('ITS Analytics System', cv2.resize(frame, (1280, 720)))  
            if cv2.waitKey(1) & 0xFF == ord('q'): break  

    finally:  
        cap.release()  
        cv2.destroyAllWindows()  
        if db_conn: db_conn.close()  

if __name__ == "__main__":  
    run_detector()  

