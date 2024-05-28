from django import forms


class UploadFile(forms.Form):
    file = forms.ImageField(label="Фотография")
