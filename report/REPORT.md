# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026 (theo ngày sửa đổi các tệp kết quả trong runtime Colab).

**Runtime Colab:** GPU (`Device: cuda`).

**Python / PyTorch / Ultralytics:** `3.13.16` / `2.11.0+cu130` / `8.4.145`.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Chưa xác minh vì không có bản notebook nguồn để so sánh; checkpoint và ngưỡng trong bản đã chạy đúng các giá trị ghi trên.

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…

   **Trả lời:** `class_id=468`; `class_name=cab`; `rank=1`; `score=0.510915`; `taxonomy_name=ImageNet-1K`. Nguồn: output ô `classification-json`, record `sample_id=traffic`.

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   **Trả lời:** Record nói về cả bức ảnh. Checkpoint phân loại nhận toàn ảnh và trả về thứ hạng các lớp; record không có box hay mask để xác định một vật thể riêng.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   **Trả lời:** Danh sách lớp đi kèm taxonomy ImageNet-1K của checkpoint `yolo11n-cls.pt`; mô hình dự đoán trong danh sách đó.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   **Trả lời:** `class_id` xác định mã lớp, `class_name` giúp đọc hiểu, còn `taxonomy_name` xác định hệ lớp đang dùng. Cần ghi cả ba để đối chiếu đúng ánh xạ mã–tên và tránh nhầm giữa các taxonomy.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   **Trả lời:** Guideline cần xác định tiêu chí chọn chủ đề chính, thứ tự ưu tiên khi có nhiều vật hoặc chủ đề, cách xử lý khi các lựa chọn ngang nhau và trường hợp phải hỏi người phụ trách.

6. Vì sao không được lấy `score` của model làm ground truth?

   **Trả lời:** `score` là mức tin cậy của mô hình, mô hình vẫn có thể đoán sai khi score cao. Ground truth phải do người gán và kiểm tra theo guideline; score không chứng minh nhãn đúng.


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

   **Trả lời:** Chọn record đầu tiên có `sample_id=kitchen`: `class_name=person`; `score=0.912625`; `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`; `bbox_width=113.58`; `bbox_height=279.68`. Nguồn: `detection_predictions.json` tải trực tiếp từ thư mục `day1_lab_outputs/` trong Colab, ngưỡng `0.35`.

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   **Trả lời:** Trên ảnh kích thước `640 × 427` pixel, box của người có góc trên-trái tại `(385.33, 69.24)` và góc dưới-phải tại `(498.92, 348.92)`, tính từ góc trên-trái ảnh; trục x tăng sang phải, trục y tăng xuống dưới. Chiều rộng là `113.58` pixel, chiều cao là `279.68` pixel theo JSON. Hiệu hai tọa độ x đã làm tròn là `113.59`, lệch `0.01` pixel so với chiều rộng vì notebook tính chiều rộng từ tọa độ gốc trước khi làm tròn.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   **Trả lời:** Trên `kitchen`, ngưỡng `0.20` cho 17 prediction; ngưỡng `0.60` cho 6 prediction. Ngưỡng tạo JSON `0.35` cho 11 prediction. Nguồn: output ô `detection-visual`.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   **Trả lời:** Reviewer phải kiểm tra thêm các hộp được giữ lại: có vật thể thật hay không, đúng lớp hay không, box có sát vật thể và bị trùng hay không; đồng thời rà soát vật thể bị bỏ sót. Không loại ground truth chỉ vì model có score thấp.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   **Trả lời:** Quy tắc đề xuất cho nhãn phần nhìn thấy: mỗi vật thể thuộc phạm vi có một box sát phần nhìn thấy; box nằm trong ảnh và không tự mở rộng để đoán phần vật thể bị che hoặc nằm ngoài ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   **Trả lời:** Trên hình `kitchen`, người ở mép trái chỉ lộ một phần và bị cắt mép ảnh. Tôi đưa trường hợp này lên hỏi Lab Coach nếu guideline chưa quy định mức nhìn thấy tối thiểu để gán người; sau khi thống nhất, bổ sung quy tắc xử lý vật bị cắt mép để áp dụng nhất quán cho nhiều ảnh.


## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

   **Trả lời:** `instance_id=kitchen-001`; `class_name=person`; `score=0.899318`; `polygon_point_count=348`; 5 điểm đầu của `polygon_xy` là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]` (đơn vị pixel). Nguồn: `segmentation_predictions.json` tải trực tiếp từ thư mục `day1_lab_outputs/` trong Colab, ngưỡng `0.35`.

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   **Trả lời:** Polygon cho biết đường biên và vùng hình dạng của từng vật thể, giúp phân biệt vật thể với phần nền nằm trong box. Box chỉ mô tả phạm vi bao quanh bằng hình chữ nhật.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   **Trả lời:** `instance_id` phân biệt từng đối tượng trong output lab: hai vật cùng lớp vẫn có ID và polygon riêng. Class ID chỉ loại vật thể; tracking ID liên kết cùng một vật qua các khung hình. ID dạng `kitchen-001` trong notebook được tạo theo số thứ tự prediction, không bảo đảm nhận dạng cùng vật giữa các lần chạy.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   **Trả lời:** Quy tắc đề xuất cho mask phần nhìn thấy: vẽ biên sát vùng vật thể nhìn thấy, tách khỏi nền và vật khác; không tự nối hoặc vẽ thêm phần bị che khi guideline không yêu cầu.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   **Trả lời:** Trên hình `kitchen`, phần người ở mép trái nằm sát các bát trên bàn; ranh giới phần tay và đồ vật khó phân biệt, người cũng bị cắt mép. Nếu guideline về mask phần nhìn thấy chưa đủ để xác định biên, tôi escalation lên Lab Coach để thống nhất, tránh tự vẽ phần bị che.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho toàn ảnh theo taxonomy và guideline. | JSON của `traffic`: top-1 `cab` (0.510915), top-2 `minibus` (0.164284); cảnh có nhiều vật nên cần tiêu chí chọn chủ đề chính, chưa thể coi top-1 là nhãn đúng. | Quan sát toàn ảnh và chọn một lớp theo guideline; hỏi khi không đủ căn cứ. | Kiểm tra lớp, taxonomy và cách áp dụng tiêu chí nhất quán. |
| Phát hiện vật thể | Một lớp và một box cho mỗi vật thể thuộc phạm vi. | PNG `kitchen`: người ở mép trái bị cắt; JSON có prediction `person`, score `0.610931`, box `[0.12, 263.18, 61.55, 310.78]` cho vùng này. | Gán lớp, vẽ box theo guideline cho phần nhìn thấy; escalation nếu chưa rõ cách xử lý vật bị cắt. | Kiểm tra đúng lớp, độ sát box, hộp trùng, vật bỏ sót và quy tắc che khuất/cắt mép. |
| Instance segmentation | Một lớp và một mask riêng cho mỗi đối tượng. | PNG `kitchen`: phần người sát các bát ở mép trái khó phân biệt biên và bị cắt mép; JSON có `kitchen-009`, lớp `person`, score `0.431438`, 65 điểm polygon cho vùng này. | Vẽ mask theo phần nhìn thấy, tách các instance; hỏi khi biên chưa rõ. | Kiểm tra biên, vùng thừa/thiếu, lớp và việc tách đúng từng instance. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   **Trả lời:** Chỉ dùng ba ảnh COCO công khai do notebook chỉ định, giữ thông tin nguồn và giấy phép, không đưa ảnh cá nhân, dữ liệu nội bộ hoặc thông tin cá nhân vào báo cáo/output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   **Trả lời:** Tôi dừng xử lý dữ liệu ngoài phạm vi và báo Lab Coach hoặc giảng viên phụ trách để nhận hướng dẫn trước khi tiếp tục.


## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [X] `REPORT.md` (file này đã điền)
- [X] `day1_lab_outputs/classification_predictions.json`
- [X] `day1_lab_outputs/detection_predictions.json`
- [X] `day1_lab_outputs/segmentation_predictions.json`
- [X] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [X] `day1_lab_outputs/visuals/classification_top5.png`
- [X] `day1_lab_outputs/visuals/detection_predictions.png`
- [X] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [X] Đã nộp **link repository** trên LMS

**Ghi chú kiểm chứng:** Nguồn là output và hình được lưu trong notebook `Bản_sao_của_day1_understand_labels.ipynb` (các ô `environment-setup`, `classification-json`, `detection-visual`, `segmentation-visual`, `validate-and-archive`) và hai tệp [detection_predictions.json](detection_predictions.json), [segmentation_predictions.json](segmentation_predictions.json) tải trực tiếp từ [notebook Colab](https://colab.research.google.com/drive/1HvM-LkYOLL6c4uZGYW-DLRolv5xbZRkK). Ô validation đã in `PASS` cho 3 JSON, 3 PNG, attribution và instance ID duy nhất; kết quả này cũng được xác nhận trên Colab đang mở. Các mục về file trên GitHub và nộp LMS chưa được đánh dấu vì chưa kiểm tra việc nộp bài.
