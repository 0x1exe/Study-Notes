
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


# 💥Spark
## Как правильно просить ресурсы / устройство Spark

Стандартная инициализация SparkSession выглядит примерно так:
```
from pyspark.sql import SparkSession

app_name = 'Your App'

conf = {
    'spark.dynamicAllocation.enabled': 'true',
	'spark.shuffle.service.enabled': 'true',
    'spark.dynamicAllocation.maxExecutors': 10,
    'spark.executor.memory': '32g',
    'spark.executor.cores': '4',
    'spark.driver.memory': '4g',

	# какие-то другие параметры
}

builder = (
    SparkSession
    .builder
    .appName(app_name)
)

for k, v in conf.items():
    builder.config(k, v)

spark = builder.getOrCreate()
```
Здесь не упомянуты все параметры, на самом деле их несколько десятков.

Управление ресурсами для Spark происходит так:
1. Приложение Spark работает в виде процессов двух типов: драйвер и экзекьютор. Драйвер всегда один, экзекьюторов много.
2. У драйвера и экзекьюторов нужно управлять ядрами и памятью, есть тонкие настройки.
3. Количеством экзекьюторов тоже нужно управлять.

Spark написан на Scala и немного на Java. Поэтому сам по себе он — это процессы, работающие в JVM (Java Virtual Machine). Но когда мы работаем на pySpark, к ним добавляются процессы-компаньоны на Python. Python используется драйвером как интерфейс для вызова функций, а экзекьюторы — для выполнения user-defined functions на Python.

Драйвер в Spark выполняет две основных задачи: планирование расчётов и сбор результатов. Для планирования расчётов дополнительные ресурсы не нужны. А вот сбор результатов стоит рассмотреть подробнее.

**Драйвер в Spark можно разворачивать двумя способами: в режиме cluster и client. cluster используется для промышленных расчетов. Ресурсы драйвера выделяются ресурсным менеджером YARN, вы можете контролировать все настройки.**

В pySpark ноутбуках используется режим client. Это значит, что драйвер запускается на сервере рядом с вашим ноутбуком, а часть параметров его настройки вам недоступна.

При работе с Python-драйвером важно помнить, что лимитами ресурсов невозможно управлять. Если вы выполняете большой запрос toPandas и запускаете тяжелую локальную ML-модель, это может занять все ресурсы сервера и повлиять на работу других процессов в кластере.

Параметрами контекста мы влияем только на JVM-процесс:

`spark.driver.cores`

Этот параметр не используется при deploy-mode=client. Задавать не нужно.

`spark.driver.memory`

Память драйвера. Если toPandas будет падать с ошибкой OutOfMemory, ограничение будет в этом параметре. Дефолт — 1 Гб. Если не используете toPandas для больших датафреймов, менять не нужно.

`spark.driver.maxResultSize`

Объём данных, который драйвер может собирать. В том числе и toPandas-ом. По умолчанию менять не нужно. Если вы изменили spark.driver.memory, нужно поднимать maxResultSize до эквивалентных значений.

`spark.driver.memoryOverhead`

Этот параметр менять не нужно. При deploy-mode=client на Python эта память не распространяется. При ошибках спарк даёт совет поменять memoryOverhead — это вводит в заблуждение. Менять нужно параметр memory. memoryOverhead выделяется как 10% от memory и этого должно быть достаточно.

- Почему нужно использовать динамическую аллокацию ресурсов

В Spark существует два типа аллокаций ресурсов — статическая и динамическая. В ноутбуках всегда рекомендуется использование динамической.

В статической аллокации вы при старте контекста задаёте количество экзекьюторов. Они закрепляются за вами, пока приложение будет работать.

**В статической аллокации вы при старте контекста задаёте количество экзекьюторов. Они закрепляются за вами, пока приложение будет работать.**

Динамическая аллокация решает эту проблему. Если вы какое-то время не пользуетесь ноутбуком, ресурсы срезаются до минимума.

Параметры для настройки динамической аллокации:

`spark.dynamicAllocation.enabled   true`

`spark.shuffle.service.enabled или spark.dynamicAllocation.shuffleTracking.enabled true`.

Нужно указать один из двух. Пробуйте первый параметр. Если будут ошибки, переключайтесь на второй.

`spark.dynamicAllocation.maxExecutors`  
Самый важный параметр. Определяет максимальное, рабочее количество экзекьюторов.

`spark.dynamicAllocation.executorIdleTimeout`  
Время, спустя которое забираются неиспользуемые экзекьюторы. Рекомендую прописывать явно. Дефолтные 60 секунд на мой взгляд — нормально.

`spark.dynamicAllocation.cachedExecutorIdleTimeout`  
Время, которое живут экзекьюторы с кешем. Если не выставить, по умолчанию будет значение бесконечность. Но администраторы вашего кластера могли изменить это значение. Хорошие аналитики активно кешируют данные в своих ноутбуках, поэтому считайте, что динамическая аллокация работать не будет. На мой взгляд, адекватное время — от 10 до 30 минут, в зависимости от спроса на ресурсы на кластере. Задаётся в секундах.

- Параметры экзекьютора 

Настройка экзекьюторов — самое сложное. Она больше всего влияет на жизнеспособность кластера. 

Сначала определитесь с размером экзекьюторов, затем настраивайте их количество.

**При определении размера экзекьютора необходимо определить баланс ядер и памяти. Аналитики часто забывают, что ресурсы выделяются на реальных серверах с конкретным соотношением ядер и оперативной памяти для YARN. Его стоит соблюдать, чтобы не получилось так, что на кластере израсходовали все ядра, а память осталась свободной.**

Чтобы понять соотношение, зайдите в YARN → Scheduler. Посмотрите общий объём ядер и памяти, поделите. Вы получите соотношение, например, 1 ядро к 7 Гб памяти. Память экзекьютора состоит из основной памяти и оверхеда. Оверхед по умолчанию равен 10%. Тогда в этом примере на 1 ядро нужно выделять 6 Гб основной памяти.

![[Pasted image 20240807095303.png]]

Плюсы больших экзекьюторов:
1. Чем больше экзекьютор, тем меньше будет межпроцессного и межсетевого трафика.
2. Чем больше экзекьютор по памяти, тем большего размера партицию он сможет обработать и не упасть.

Минусы:
1. Большие экзекьюторы сложно выделяются. Если кластер сильно загружен, ярну придётся поискать свободные места на нодах, чтобы выделить большие экзекьюторы.
2. На экзекьюторах с памятью больше 64Гб неэффективно происходит очистка памяти (garbage collection).

По объёму памяти — берите память в 5-10 раз больше, чем объём читаемых данных. Скорее всего, вам этого хватит. С меньшим объемом памяти тоже можно работать, но возможны ошибки.
По ядрам все просто: чем их больше, тем быстрее идут расчёты. Технических ограничений тут нет. Какая скорость приемлемая — вопрос сложный. Тут нужно договариваться и выстраивать стратегию.

- Общая стратегия


1. В вашей команде должен быть установлен минимальный конфиг. Если аналитики будут вести под ним большую часть работы, то 2/3 кластера будут свободны.
2. Если у вас появилась вычислительно сложная задача, перезапустите сессию с большим количеством ресурсов. Если будут злоупотребления — установите лимит.
3. Придерживайтесь хорошего тона и берите ресурсы, когда они действительно нужны. Например, на большой расчет. Разрабатывать витрину можно и на небольшом семпле данных.

##  Spark Data Caching, Repartition, Coalesce

- **Spark memory pools**

	- **Storage Pool:** Used for caching DataFrames.
	- **Execution Pool:** Used for DataFrame computations.

Benefits of caching dataframes:
	1. Reduced data processing for reused DFs
	2. Improved performance for iterative algorithms
	
- `cache()`: Simple caching with default `MEMORY_AND_DISK` storage level.
- `persist()`: Offers flexibility to customize storage level.

- **Understanding storage levels**

- Spark provides options for caching data in memory, disk, or off-heap memory (if configured).
- Serialized vs. Deserialized Formats:
- Serialized: Takes less space but requires deserialization before use (CPU overhead).
- Deserialized: Ready for use but consumes more memory.
- Replication Factor: Number of copies of cached data (increases fault tolerance but uses more memory).
- `unpersist()` removes data from the cache.

- **Repartitioning:**

- Creates a new DataFrame with a specified number of partitions.
- Uses a hash function by default to distribute rows across partitions (uniform size).
- Optionally, repartitions on one or more columns (not guaranteed uniform size).
- Triggers a shuffle operation (expensive) to redistribute data.
- Number of partitions controlled by `spark.sql.shuffle.partitions` configuration (configurable).
- Improve parallelism for operations that filter on specific columns (repartition on those columns).
- Balance data distribution across the cluster (especially with few or skewed partitions).
- Create uniform partition sizes for some operations.

- **Coalescing:**

- Reduces the number of partitions in a DataFrame.
- Merges local partitions on the same worker node without shuffling.
- Does not work for increasing partitions (use `repartition()`).
- Can lead to skewed partitions if reduction is drastic.
- -Reduce resource usage when working with DataFrames that have more partitions than needed.
- Optimize downstream operations by reducing the number of tasks.