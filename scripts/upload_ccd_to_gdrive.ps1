# Script đẩy dữ liệu datasets\ccd lên Google Drive bằng Rclone đa luồng
# Đường dẫn nguồn và đích
$source = "D:\AI\datasets\ccd"
$destination = "gdrive:datasets/ccd"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Bắt đầu upload: $source -> $destination" -ForegroundColor Green
Write-Host "Tối ưu: 16 luồng song song, chunk size 64MB, resume tự động" -ForegroundColor Yellow
Write-Host "=========================================================" -ForegroundColor Cyan

# Cập nhật PATH trong phiên làm việc
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Chạy lệnh Rclone copy
rclone copy "$source" "$destination" `
    --transfers 16 `
    --checkers 16 `
    --drive-chunk-size 64M `
    --fast-list `
    -P

Write-Host "`nĐã hoàn thành upload dữ liệu lên Google Drive!" -ForegroundColor Green
