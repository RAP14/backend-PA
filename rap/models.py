from django.db import models

class Coins(models.Model):
    name = models.CharField('NamaCoin', max_length=100)
    price = models.DecimalField('Low Price', decimal_places=2, max_digits=8)
    created = models.DateTimeField('Datetime')

# Create your models here.