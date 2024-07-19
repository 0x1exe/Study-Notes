
#   🚀 Django

## Project initiation
1. Create a virtualenv and activate(!) it
2. pip install django
3. django-admin startproject [project_name]
4. python manage.py runserver
5. python manage.py startapp [app_name]

models -> database handling
views -> urls routing handling

**Add base.apps** to INSTALLED_APPS

## Adding urls and views
### Basic views


```
from django.http import HttpResponse 

def home(request):
	return HttpResponse("Home page")
```
Then add this to urlpatterns
```
path('',home)
```
but it would be better to move **the page views into the views.py folder accordingly**.
Then create **urls.py** in our app -> from django.urls import path. In project urls.py import include and:
```
path('',include(base.urls))
```
We can pass a specific values into urls such as:
```
path('room/<str:pk>/',views.room,name='room')
```
and in **view** pass in another parameter
```
def room(request,pk)
```
add a href to a tag in html with
```
<a href = '/room/{{room.id}}/'>
```
of better, href can be:
```
href= "{%url 'room' room.id%}" 
```
If we defined room in urlpatterns
### CRUD operations
add new url for each operation:
```
path('create-room/',views.createRoom,name='create-room')
```
and then a new template for form submission
```
{% extends 'main.html'%} {% block content %}

<div>
  <form method="POST" action="">
    {%csrf_token%} {{form.as_p}}
    <input type="submit" value="Submit" />
  </form>
</div>
{%endblock content%}
```
Then we need to create a forms.py:
```
from django.forms import ModelForm
from .models import Room


class RoomForm(ModelForm):
    class Meta:
        model = Room
        fields = '__all__'
```
Navigate to views:
```
def createRoom(request):
    form = RoomForm()
    if request.method == "POST":
        form = RoomForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('home')
    context = {'form':form}
    return render(request,'base/room_form.html',context)
```
for an update Room:
```
def updateRoom(request,pk):
    room = Room.objects.get(id=int(pk))
    form = RoomForm(instance=room)

    if request.method == 'POST':
        form=RoomForm(request.POST,instance=room)
        if form.is_valid():
            form.save()
            return redirect('home')
    context = {'form':form}
    return render(request,'base/room_form.html',context)
```
for delete operation:
```
{%extends 'main.html'%} {%block content%}
<form method="POST" action="">
  {%csrf_token%}
  <p>Are you sure you want to delete "{{obj}}" ?</p>
  <a href="{{request.META.HTTP_REFERER}}">Return</a>
  <input type="submit" value="Confirm" />
</form>
{%endblock content%}
```
and view:
```
def deleteRoom(request,pk):
    room = Room.objects.get(id=int(pk))
    if request.method == "POST":
        room.delete()
        return redirect('home')
    context={'obj':room}
    return render(request,'base/delete.html',context)
```
## HTML templates 
1. Create a separate templates folder 
2. Add templates folder to settings.py like this: 'DIRS': [BASE_DIR/'templates']

Now in views we can import django.shortcuts.render  and using the following syntax:
```
render(request,html)
```
render the following page.
`render` can also accept a variable such as:
```
render(request,html,{'rooms':rooms})
```
to further address it in an html template.

## Template tags
We can iterate over an argument passed in render such as:
```
{%for room in rooms%}
<div>
	<h5>{{room.id}}: {{room.name}}</h5>
</div>
{%endfor%}
```
### Template inheritance
Templates can inherit from one another.
to inherit from an html file -> 
```
{%include 'parent.html'%}
```
Instead, we can include the following line of code into our main html to tell it. where should the child pages go
```
{%block content%}
{%endblock%}
```
Then go to child html and add:
```
{%extends 'main.html'%}

{%block content%} ## Shows where should the child content be in the main template

<h1>HOME PAGE</h1>

{%endblock%}
```


## Models

### Simple in-built sqlite database
**We could override an id to a custom one**
When we add a new model - we need to migrate:
```
python manage.py makemigrations
python manage.py migrate
```
We need to create a user to be able to access the database and such:
```
python manage.py createsuperuser
```
Inherit from django models!
```
from django.db import models
class Room(models.Model):
	host=
	topic= 
	name = models.CharField(max_length=200)
	description = models.TextField(null=True,blank=True)
	participants = 
	updated = models.DateTimeField(auto_now=True)
	created = models.DateTimeField(auto_now_add=True)
	class Meta:
       ordering = ['-updated','-created']

	def __str__(self):
		return self.name

```
Different types have different attributes, be aware of those.

We need to navigate to admin.py to make Room model visible
```
from .models import Room
admin.site.register(Room)
```
Now to address the model in our code we need to change our views a bit
```
def home(request):
    rooms = Room.objects.all()
    context = {'rooms':rooms}
    return render(request,'base/home.html',context)
```
Django has **in-built User models , which we can use**:
```
user = models.ForeignKey(User,on_delete=models.CASCADE)
```
The above code belongs to `class Message(models.Model)`


# Libs

1. bm25s
2. RAPIDS
3. Cognita
