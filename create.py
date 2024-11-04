import qrcode

def create_qr_code(data, filename="qr_code.png"):
    # Create a QR code with specified data
    qr = qrcode.QRCode(
        version=1,  # Controls the size of the QR code
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
        box_size=10,  # Size of each box in the QR code
        border=4  # Border size
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Generate image of QR code
    img = qr.make_image(fill="black", back_color="white")
    img.save(f"{filename}")
    print(f"QR code saved as {filename}")

# Generate a QR code with sample data
create_qr_code("https://www.lab2m.com", "l2m_qr.png")
