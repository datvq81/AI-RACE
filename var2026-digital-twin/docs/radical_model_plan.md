# Radical model plan

Tài liệu handoff cho GPT-5.6-sol. Thực hiện **một task mỗi lần**, kiểm tra xong
mới chuyển task tiếp theo.

## Bối cảnh cố định

- Stack ổn định: Nerfstudio `1.1.4`, gsplat `1.0.0`.
- Không nâng package trong môi trường submit hiện tại.
- Baseline sản xuất: D1b/F0, HCM0421 40k local score `72.77714`.
- H0 20k `72.29030`, khớp D1b 20k `72.29640`; control đã hợp lệ.
- Diagnostic:
  - shift/blur oracle gần `0`;
  - affine RGB nhỏ;
  - 20% edge giữ `53.15%` lỗi HCM và `47.61%` lỗi bonsai.
- Mục tiêu: cải thiện trung bình HCM0421 + bonsai ít nhất `+0.50`.
- Không đưa method mới vào `auto_pipeline.py` trước khi qua gate hai scene.

## Quy tắc thực hiện

1. Giữ nguyên pipeline D1b và các checkpoint cũ.
2. Radical code/environment nằm trong namespace hoặc container riêng.
3. Mỗi treatment phải có same-code control, cùng scene/seed/iterations.
4. Lưu command, config, metrics, Gaussian count, peak VRAM và log.
5. Không thay đồng thời density control, loss và appearance trong một ablation.

---

## R0 — Khóa baseline và harness

**Mục tiêu:** bảo đảm mọi radical method dùng cùng split, camera và evaluator.

**Việc làm**

- Tạo `configs/experiments/radical/`.
- Tạo adapter dùng lại:
  - `data/local_validation/<scene>`;
  - `scripts/render_test_pose.py` hoặc renderer tương đương;
  - `scripts/evaluate_local_validation.py`.
- Ghi manifest gồm git commit, environment, scene, seed và command.

**Pass**

- Render baseline radical-control có đúng số ảnh và scorer chạy không sửa đổi.
- Chênh control lặp lại không quá `0.10 Score`.

---

## R1 — Pixel-GS density control

**Mục tiêu:** thay gradient trung bình theo view bằng pixel-aware gradient.

**Việc làm**

- Đọc paper và official implementation Pixel-GS.
- Thu thập cho mỗi Gaussian:
  - absolute screen-space gradient;
  - số pixel/diện tích screen-space mà Gaussian phủ;
  - khoảng cách camera.
- Dùng pixel coverage để weighted-average gradient qua các view.
- Có distance scaling để hạn chế floater gần camera.
- Thêm control `P0` và treatment `P1`; chưa thêm frequency loss.

**Pass**

- Unit test weighting bằng tensor tổng hợp.
- Smoke train đi qua ít nhất hai refinement cycles.
- `P1 - P0 >= +0.20` tại HCM0421 20k mới chuyển R2.

---

## R2 — FreGS progressive frequency loss

**Mục tiêu:** phục hồi high-frequency detail thay vì chỉ tăng số Gaussian.

**Việc làm**

- Thêm FFT loss tách low/high-frequency giữa render và GT.
- Schedule coarse-to-fine; high-frequency chỉ tăng sau full resolution.
- Tạo:
  - `FREQ0`: Pixel-GS control;
  - `FREQ1`: low + high frequency mức bảo thủ;
  - `FREQ2`: treatment mạnh nếu FREQ1 ổn định.
- Không đổi density threshold/culling trong task này.

**Pass**

- Loss hữu hạn, không NaN, không tạo autograd graph ở target.
- `FREQ1 - FREQ0 >= +0.30` trên HCM0421 hoặc dừng nhánh.
- Winner phải không giảm quá `0.10` trên bonsai.

---

## R3 — MCMC-GS branch độc lập

**Mục tiêu:** thay hoàn toàn split/clone/cull heuristic.

**Việc làm**

- Tạo environment/container riêng dùng official 3DGS-MCMC hoặc modern gsplat.
- Viết loader cho transforms/local-validation hiện tại.
- Cố định Gaussian budgets: nhỏ, vừa, lớn.
- Chạy `M0` official default và `M1` budget lớn; chưa trộn Pixel-GS.

**Pass**

- Renderer giữ nguyên filename/resolution/camera.
- So sánh macro-score HCM0421 + bonsai với winner R2.
- Chỉ giữ MCMC nếu macro gain ít nhất `+0.30`.

---

## R4 — Dense initialization bằng VGGT

**Mục tiêu:** thay COLMAP sparse seed bằng dense multi-view geometry.

**Việc làm**

- Chạy VGGT theo các camera chunks có overlap.
- Align point maps vào COLMAP bằng Sim(3).
- Lọc point bằng confidence và multi-view consistency.
- Voxel-downsample ở nhiều mức; giữ thêm point quanh image edges.
- Khởi tạo winner R2/R3 bằng dense cloud.

**Pass**

- Không thay camera chính thức.
- Có visualization point cloud và thống kê point theo confidence.
- Dense init phải hơn sparse init `>= +0.20` ở hai scene.

---

## R5 — Cross-fitted neural refiner

**Mục tiêu:** học sửa residual có hệ thống mà không dùng hidden GT.

**Việc làm**

- Giữ một outer validation chưa dùng cho refiner.
- Tạo inner folds; train base renderer bỏ từng fold và render out-of-fold.
- Dataset refiner:

```text
input  = RGB render + alpha + depth + edge/confidence
target = ground-truth RGB
output = bounded RGB residual
```

- Dùng U-Net/Restormer nhỏ, patch training và tiled full-HD inference.
- Loss gồm L1 + SSIM + LPIPS; đánh giá vẫn dùng scorer chuẩn.

**Pass**

- Outer validation chưa xuất hiện trong base/refiner training.
- Refiner tăng macro-score hai scene `>= +0.50`.
- Không đổi kích thước, filename hoặc số ảnh output.

---

## R6 — Multi-renderer fusion

**Mục tiêu:** kết hợp lỗi bổ sung của GS và radiance field.

**Việc làm**

- Train Zip-NeRF hoặc renderer continuous-grid trên cùng camera.
- Render GS winner và Zip-NeRF cho các out-of-fold views.
- Refiner R5 nhận thêm RGB/depth/confidence từ cả hai renderer.
- A/B:
  - best single renderer;
  - average RGB;
  - learned fusion.

**Pass**

- Learned fusion phải hơn best single model `>= +0.30`.
- Xác nhận trên scene thứ ba trước khi train toàn bộ 7 scene.

---

## R7 — Chỉ khi các nhánh trên chưa đủ

Thử lần lượt, không trộn ngay:

1. Scaffold-GS/FeatSplat thay SH bằng feature + view-conditioned decoder.
2. PGSR/2D surfel cho mái và pin mặt trời.
3. Mixed primitives: surfel cho mặt phẳng, line/needle Gaussian cho dây/thép,
   3D Gaussian cho background.
4. Semantic mask chỉ dùng tăng **densification priority**, không dùng masked
   loss 10× và không bỏ background.

## Gate cuối

Chỉ chạy 40k/toàn bộ scene khi một candidate:

- thắng control trên ít nhất hai scene;
- macro gain `>= +0.50`;
- thắng hoặc hòa ở cả PSNR, SSIM và LPIPS trade-off hợp lý;
- không phụ thuộc local-GT correction lúc inference;
- qua `scripts/check_submission.py`.

## Thứ tự ưu tiên

```text
R0 → R1 Pixel-GS → R2 FreGS
                  ↘ R3 MCMC (branch riêng)
winner → R4 dense init → R5 refiner → R6 fusion
R7 chỉ là moonshot cuối
```
