import commune as c
# This code example demonstartes how to generate a QR code from Text.
# Initialize the BarcodeGenerator
# Specify Encode type

class Barcode(c.Module):
    def text2qrcode(self, text='whadup', filename='barcode.png'):
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(text)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img.save(filename)
        return filename
    
    def install(self):
        c.cmd('pip3 install qrcode[pil]', verbose=True)


