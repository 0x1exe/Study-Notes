
# Database solutions
![[Databases 2.jpg]]

![[Databases 2 1.jpg]]



# Network protocols
## HTTP

HTTP is a client-server protocol: requests are sent by one entity, the user-agent (or a proxy on behalf of it). Most of the time the user-agent is a Web browser, but it can be anything, for example, a robot that crawls the Web to populate and maintain a search engine index.

Each individual request is sent to a server, which handles it and provides an answer called the _response_. Between the client and the server there are numerous entities, collectively called proxies, which perform different operations and act as gateways or caches, for example.

In reality, there are more computers between a browser and the server handling the request: there are routers, modems, and more. Thanks to the layered design of the Web, these are hidden in the network and transport layers. HTTP is on top, at the application layer. Although important for diagnosing network problems, the underlying layers are mostly irrelevant to the description of HTTP.

-  HTTP is stateless, but not sessionless.
	HTTP is stateless: there is no link between two requests being successively carried out on the same connection. This immediately has the prospect of being problematic for users attempting to interact with certain pages coherently, for example, using e-commerce shopping baskets. But while the core of HTTP itself is stateless, HTTP cookies allow the use of stateful sessions. Using header extensibility, HTTP Cookies are added to the workflow, allowing session creation on each HTTP request to share the same context, or the same state.
- HTTP connection
	A connection is controlled at the transport layer, and therefore fundamentally out of scope for HTTP. HTTP doesn't require the underlying transport protocol to be connection-based; it only requires it to be _reliable_, or not lose messages (at minimum, presenting an error in such cases). Among the two most common transport protocols on the Internet, TCP is reliable and UDP isn't. HTTP therefore relies on the TCP standard, which is connection-based.
- Features controllable with HTTP
	- _[Caching](https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching)_: How documents are cached can be controlled by HTTP. The server can instruct proxies and clients about what to cache and for how long. The client can instruct intermediate cache proxies to ignore the stored document.
	- _Relaxing the origin constraint_: To prevent snooping and other privacy invasions, Web browsers enforce strict separation between websites. Only pages from the **same origin** can access all the information of a Web page. Though such a constraint is a burden to the server, HTTP headers can relax this strict separation on the server side, allowing a document to become a patchwork of information sourced from different domains; there could even be security-related reasons to do so.
	- _Authentication_: Some pages may be protected so that only specific users can access them. Basic authentication may be provided by HTTP, either using the [`WWW-Authenticate`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/WWW-Authenticate) and similar headers, or by setting a specific session using [HTTP cookies](https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies).
	- _[Proxy and tunneling](https://developer.mozilla.org/en-US/docs/Web/HTTP/Proxy_servers_and_tunneling)_: Servers or clients are often located on intranets and hide their true IP address from other computers. HTTP requests then go through proxies to cross this network barrier. Not all proxies are HTTP proxies. The SOCKS protocol, for example, operates at a lower level. Other protocols, like ftp, can be handled by these proxies.
	- _Sessions_: Using HTTP cookies allows you to link requests with the state of the server. This creates sessions, despite basic HTTP being a state-less protocol. This is useful not only for e-commerce shopping baskets, but also for any site allowing user configuration of the output.
## TCP
TCP meaning Transmission Control Protocol, is a communications standard for delivering data and messages through networks. 
TCP is a basic standard that defines the rules of the internet and is a common protocol used to deliver data in digital network communications.

The set of TCP/IP protocols:

![[Pasted image 20240311151025.png]]
В протоколе TCP/IP строго зафиксированы правила передачи информации от отправителя к получателю. Сообщение или поток данных приложения отправляется протоколу Internet транспортного уровня, то есть **Протоколу пользовательских дейтаграмм** ( **UDP**) или **Протоколу управления передачей** (**TCP**). Получив данные от приложения, эти протоколы разделяют всю информацию на небольшие блоки, которые называются _пакетами_. К каждому пакету добавляется адрес назначения, а затем пакет передается на следующий уровень протоколов Internet, то есть сетевой уровень

На сетевом уровне пакет помещается в дейтаграмму **протокола Internet** (**IP**), к которой добавляется заголовок и концевик. Протокол сетевого уровня определяет адрес следующего пункта назначения IP-дейтаграммы (она может быть передана сразу получателю или на промежуточный шлюз) и отправляют ее на уровень сетевого интерфейса.

Уровень сетевого интерфейса принимает **IP**-дейтаграммы и передает их в виде _кадров_ с помощью аппаратного обеспечения, такого как адаптер Ethernet или Token-Ring.

Transfer of information from client to server with TCP protocol: 
![[Pasted image 20240311151220.png]]
The TCP header ensures error control through various mechanisms and procedures implemented to detect, manage, and recover from errors that may occur during data transmission over a network

1. **Sequence Numbers**: TCP assigns a unique sequence number to each segment of data it sends, aiding in the proper ordering of received data at the destination
2. **Acknowledgements (ACKs)**:TCP uses acknowledgment messages to confirm the successful receipt of data. If the sender doesn’t receive an acknowledgment within a specified time, it assumes that an error has occurred, triggering retransmission
3. **Checksums**: TCP employs a checksum algorithm to create a numerical value that represents the contents of a packet. The receiver calculates its own checksum upon receiving the packet and compares it with the sender’s checksum to detect errors
4. **Retransmission**: When TCP detects a lost or corrupted packet (through timeouts or duplicate acknowledgment signals), it initiates a retransmission of the affected data to ensure its successful delivery to the receiver

The TCP header consists of various fields that play specific roles in managing data transmission. Here is an example of a TCP protocol header with its key fields:
- **Source Port**: A 16-bit field specifying the port number of the application sending the data.
- **Destination Port**: Another 16-bit field indicating the port number of the application receiving the data.
- **Sequence Number**: A 32-bit field that assigns a unique sequence number to each byte of data in the TCP segment.
- **Acknowledgement Number**: Also a 32-bit field specifying the sequence number of the next expected byte by the receiver.
- **Data Offset (HLEN)**: A 4-bit field indicating the size of the TCP header in 32-bit words.
- **Control Flags**: A 6-bit field containing various control flags like URG, ACK, PSH, RST, SYN, and FIN.
- **Window Size**: A 16-bit field specifying the size of data that can be sent without acknowledgment.
- **Checksum**: A 16-bit field used for error detection by checksumming data, header, and pseudo-header.
- **Urgent Pointer**: A 16-bit field indicating where urgent data ends, used to deliver urgent data quickly

## UDP 
**User Datagram Protocol (UDP)** is a communication protocol primarily used for establishing low-latency and loss-tolerating connections between applications. Unlike TCP, UDP is connectionless and unreliable, meaning it does not require establishing a connection before data transfer and does not guarantee data delivery or offer features like retransmission of lost or corrupted messages

**Key Points about UDP:**
- **Speed**: UDP speeds up transmissions by enabling data transfer before the receiving party provides an agreement, making it beneficial for time-sensitive communications like VoIP, DNS lookup, and media streaming
- **Process-to-Process Communication**: UDP enables process-to-process communication, in contrast to TCP which supports host-to-host communication
- **Header Composition**: The UDP header consists of fields like source port, destination port, length, and checksum, with no error control or flow control provided by UDP itself
- **Applications**: UDP is used in applications that can tolerate lost data such as gaming, voice or video conferencing, media streaming, and real-time protocols like DNS or NTP
- **No congestion control** : UDP does not prevent network congestion. Congestion occurs when too many packets are sent into a single link or node in the network. It could lead to packet loss, queueing delays or even blocking or slowing of new connections.
- **Low overhead**

The difference between TCP and UDP in terms of message delivery lies in their fundamental design and functionality:

- **TCP (Transmission Control Protocol)**: TCP is a connection-oriented protocol that guarantees reliable data delivery by establishing a connection before transmitting data. It ensures data is delivered in the correct order through mechanisms like packet sequencing and acknowledgment. TCP is slower than UDP due to its focus on reliability and error checking, making it suitable for applications where data integrity is crucial, such as file transfers, emails, and web browsing
- **UDP (User Datagram Protocol)**: UDP, on the other hand, is a connectionless protocol that does not require a connection setup before data transmission. It does not guarantee delivery or order of packets, making it faster and more efficient than TCP. UDP is preferred for real-time applications like gaming, video streaming, VoIP, and broadcasting where speed is prioritized over reliability. While UDP lacks error checking and retransmission of lost packets like TCP, it excels in scenarios where some data loss can be tolerated without significant impact on the user experience

- UDP header consists of: 
	1. **Source port(16 bits)**
	2. **Destination port(16 bits)**
	3. **Length (16 bits)** : Specifies the length in bytes of the UDP header and data
	4. **Chekcum (16 bits)**: Optional field used for error checking of the header and data




## WebRTC
### Пример из практики: онлайн школа танцев

Пара слов о проекте, в котором мы использовали WebRTC. Нам пришел запрос на разработку приложения для онлайн школы танцев. Стандартная группа для каждого урока — 16 пользователей (один учитель и 15 учеников).

Одна из сложнейших задач проекта — добиться идеальной синхронизации 15 видеопотоков для студентов.

Проблема синхронизации возникала из-за того, что у каждого пользователя разная скорость соединения, местоположение и интернет-провайдер. Поэтому мы развернули медиа-сервер [_Wowza_](https://www.wowza.com/), который собрал все видеопотоки. Затем мы разместили медиа-сервер и веб-сайт приложения на Amazon, что снизило нагрузку на пользовательские устройства. Расчеты, обработка, синхронизация и мультиплексирование видеопотоков выполняются на сервере — учитель и ученики получают материалы, готовые к воспроизведению.

Синхронизация была достигнута с помощью [_FFmpeg_](https://www.ffmpeg.org/) — инструмента, который позволяет гибко и удобно управлять передачей аудио и видео потоков.

Нам нужно было найти решение проблемы отображения видеопотоков без использования сторонних систем. Мы решили использовать технологию WebRTC, и это оказалось идеальным решением для потоковой передачи видео через браузер.

## RTMP
**RTMP (Real-Time Messaging Protocol)** is a communication protocol designed for streaming audio, video, and data over the Internet. Originally developed by Macromedia, RTMP facilitates low-latency communication by maintaining persistent connections and splitting streams into fragments for efficient transmission. 

Some common applications of RTMP (Real-Time Messaging Protocol) include:
1. **Live Video Streaming**: RTMP is widely used for live video streaming to social media networks, media servers, and live streaming platforms over the internet. It ensures real-time transfer of video data without significant delays or buffering, allowing global audiences to experience live events, webinars, or social media broadcasts seamlessly
2. **Interactive Applications**: RTMP supports interactive, multiparty applications with low-latency communication. It enables seamless integration of live or recorded video, broadcast streaming, and video-on-demand use cases. This makes it suitable for applications requiring real-time interactions between users, such as live Q&A sessions or interactive broadcasts
3. **Secure Data Transmission**: RTMPS (RTMP over TLS/SSL) is widely used to ensure secure video data transmission by encrypting the data, adding an extra layer of security to prevent unauthorized access or breaches. Industries handling sensitive information like healthcare, finance, and government agencies commonly utilize RTMPS for secure data transfer
4. **Audio and Video Inputs**: RTMP is compatible with specific audio and video inputs like AAC (Advanced Audio Codec), AAC-LC (Low Complexity), HE-AAC+ (High-Efficiency Advanced Audio Codec) for audio, and H.264 for video. These encoding options provide flexibility and optimization for various streaming scenarios based on quality and efficiency requirements

RTMP (Real-Time Messaging Protocol) ensures low-latency for interactive, multiparty applications through several key mechanisms:

1. **Multiplexed Streams**: RTMP enables each stream to be multiplexed, allowing small high-frequency audio messages to be interleaved through multiple chunks representing a larger video frame. This construction permits a low-power client to prioritize audio playback and drop video data, optimizing resource usage and maintaining low latency.
2. **Reliable Transport**: When using a reliable, ordered transport protocol like TCP/IP, RTMP splits larger messages into smaller "chunks" for multiplexed delivery. This chunking mechanism reduces overhead and allows small messages to be sent efficiently, contributing to low-latency communication in interactive applications
3. **Stream Construction**: RTMP supports multiple synchronized audio, video, and data streams, enhancing the flexibility and efficiency of data transmission. While some client or server implementations may limit the number or type of streams, RTMP's ability to handle various streams contributes to maintaining low latency in multiparty applications

**multiplexing is a method by which multiple analog or digital signals are combined into one signal over a shared medium. The aim is to share a scarce resource – a physical transmission medium**

1. **Channel Sharing**: Multiplexing in RTMP allows different data types, such as audio, video, and other messages, to share the same connection. This way, separate streams of data can be sent simultaneously without the need for individual channels for each data type.
2. **Streamlining Bandwidth**: By sending multiple streams over a single connection, multiplexing optimizes the use of available bandwidth. It minimizes the overhead associated with opening multiple connections and reduces latency, making the streaming process more fluid and efficient.
3. **Message Fragmentation**: RTMP deals with messages that could be larger than the network's supported packet size. It fragments messages into smaller chunks that can be interleaved with chunks from other streams. This ensures continuous data flow and makes it possible to prioritize certain types of messages (like keyframes in video) to maintain stream quality.
4. **Prioritization and Control**: Multiplexing also allows RTMP to prioritize certain streams or messages, providing better control over the delivery of content. For example, ensuring that keyframes or audio signals are delivered on time for smooth playback, even if that means dropping non-essential data.
5. **Error Correction Mechanisms**: RTMP supports basic error correction methods that work well with multiplexed streams. Acknowledgment packets help to confirm receipt of data and facilitate the retransmission of lost packets without affecting other multiplexed streams.
7. **Session Control**: RTMP allows the server to control sessions interactively, even while multiplexed data is being transmitted. This includes managing flow control, adjusting to network conditions in real-time, and providing feedback mechanisms that are crucial for maintaining stream quality.

## MPEG-DASH

Streaming is a way of delivering data over the Internet so that a device can start displaying the data before it fully loads. Video is streamed over the Internet so that the client device does not have to download the entire video file before playing it.

MPEG-DASH is a streaming method. DASH stands for "Dynamic Adaptive Streaming over HTTP. Because it is based on HTTP, any origin server can be set up to serve MPEG-DASH streams.

MPEG-DASH is similar to HLS, another streaming protocol, in that it breaks videos down into smaller chunks and encodes those chunks at different quality levels. This makes it possible to stream videos at different quality levels, and to switch in the middle of a video from one quality level to another one.
The main steps in the MPEG-DASH streaming process are:

1. **Encoding and segmentation:** The origin server divides the video file into smaller segments a few seconds in length. The server also creates an index file – like a table of contents for the video segments. Then the segments are encoded, meaning formatted in a way that multiple devices can interpret. MPEG-DASH allows the use of any encoding standard.
2. **Delivery:** When users start watching the stream, the encoded video segments are pushed out to client devices over the Internet. In almost all cases, a content delivery network (CDN) helps distribute the stream more efficiently.
3. **Decoding and playback:** As a user's device receives the streamed data, it decodes the data and plays back the video. The video player automatically switches to a lower or higher quality picture in order to adjust to network conditions – for example, if the user currently has very little bandwidth, the video will play at a lower quality level that uses less bandwidth.

[Adaptive bitrate streaming](https://www.cloudflare.com/learning/video/what-is-adaptive-bitrate-streaming/) is the ability to adjust video quality in the middle of a stream as network conditions change. Several streaming protocols, including MPEG-DASH, HLS, and HDS, allow for adaptive bitrate streaming.

Adaptive bitrate streaming is possible because the origin server encodes video segments at several different quality levels. This happens during the encoding and segmentation processes. A video player can switch from one quality level to another one in the middle of the video without interrupting playback. This prevents the video from stopping altogether if network bandwidth is suddenly reduced.

LS is another streaming protocol in wide use today. MPEG-DASH and HLS are similar in a number of ways. Both protocols run over HTTP, use [TCP](https://www.cloudflare.com/learning/ddos/glossary/tcp-ip/) as their transport protocol, break video into segments with an accompanying index file, and offer adaptive bitrate streaming.

However, several key differences distinguish the two protocols:

**Encoding formats:** MPEG-DASH allows the use of any encoding standard. HLS, on the other hand, requires the use of [H.264](https://www.cloudflare.com/learning/video/what-is-h264-avc/) or H.265.

**Device support:** HLS is the only format supported by Apple devices. iPhones, MacBooks, and other Apple products cannot play video delivered over MPEG-DASH.

**Segment length:** This was a larger difference between the protocols before 2016, when the default segment length for HLS was 10 seconds. Today the default length for HLS is 6 seconds, although it can be adjusted from the default. MPEG-DASH segments are usually between 2 and 10 seconds in length, although the optimum length is 2-4 seconds.

**Standardization:** MPEG-DASH is an international standard. HLS was developed by Apple and has not been published as an international standard, even though it has wide support.

LS is another streaming protocol in wide use today. MPEG-DASH and HLS are similar in a number of ways. Both protocols run over HTTP, use [TCP](https://www.cloudflare.com/learning/ddos/glossary/tcp-ip/) as their transport protocol, break video into segments with an accompanying index file, and offer adaptive bitrate streaming.

However, several key differences distinguish the two protocols:

**Encoding formats:** MPEG-DASH allows the use of any encoding standard. HLS, on the other hand, requires the use of [H.264](https://www.cloudflare.com/learning/video/what-is-h264-avc/) or H.265.

**Device support:** HLS is the only format supported by Apple devices. iPhones, MacBooks, and other Apple products cannot play video delivered over MPEG-DASH.

**Segment length:** This was a larger difference between the protocols before 2016, when the default segment length for HLS was 10 seconds. Today the default length for HLS is 6 seconds, although it can be adjusted from the default. MPEG-DASH segments are usually between 2 and 10 seconds in length, although the optimum length is 2-4 seconds.



# SQL
## Problem 1
```
+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| id            | int     |
| recordDate    | date    |
| temperature   | int     |
+---------------+---------+
id is the column with unique values for this table.
There are no different rows with the same recordDate.
This table contains information about the temperature on a certain day.

Write a solution to find all dates' `Id` with higher temperatures compared to its previous dates (yesterday).
```

Solution: 
```
with prev_temp as (
    select *,
    lag(temperature) over (order by recordDate) as prev_temp,
    lag(recordDate) over (order by recordDate) as prev_date
    from Weather
)
select id from prev_temp where temperature > prev_temp and recordDate - prev_date = 1
```
Or using join:
```
select t1.id from Weather t1
join Weather t2 on t1.temperature > t2.temperature
and t1.recordDate - t2.recordDate = 1
```
## Problem 2
Problem:
```
There is a factory website that has several machines each running the **same number of processes**. Write a solution to find the **average time** each machine takes to complete a process.

The time to complete a process is the `'end' timestamp` minus the `'start' timestamp`. The average time is calculated by the total time to complete every process on the machine divided by the number of processes that were run.

The resulting table should have the `machine_id` along with the **average time** as `processing_time`, which should be **rounded to 3 decimal places**.

Return the result table in **any order**.
```

Solution: 
```
with process_start as (select * from Activity where activity_type='start'),
process_end as (select * from Activity where activity_type ='end')
  

select process_start.machine_id,
round(cast(avg(process_end.timestamp - process_start.timestamp) as numeric),3) as processing_time
from process_start
left join process_end on
process_start.machine_id = process_end.machine_id  
and process_start.process_id = process_end.process_id
group by process_start.machine_id
```
One trick to avoid long type-casting is to replace `cast(avg(process_end.timestamp - process_start.timestamp) as numeric)` with `avg(process_end.timestamp - process_start.timestamp) :: NUMERIC`


## Problem 3
```
Write a solution to find the number of times each student attended each exam.

Return the result table ordered by `student_id` and `subject_name`.
```
Solution №1:
```
with template as (
select * from Students
cross join Subjects
order by student_id,subject_name),

count_exams as
(select student_id,subject_name,count(*) as attended_exams
from Examinations
group by student_id,subject_name)

select template.*,
coalesce(count_exams.attended_exams,0) as attended_exams  
from template
left join count_exams 
on count_exams.student_id=template.student_id
and count_exams.subject_name=template.subject_name
```
Solution №2 (a quite faster one):
```
SELECT s.student_id, s.student_name, sub.subject_name, COUNT(e.student_id) AS attended_exams
FROM Students s
CROSS JOIN Subjects sub
LEFT JOIN Examinations e ON s.student_id = e.student_id AND sub.subject_name = e.subject_name
GROUP BY s.student_id, s.student_name, sub.subject_name
ORDER BY s.student_id, sub.subject_name
```
## Problem 4
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| name        | varchar |
| department  | varchar |
| managerId   | int     |
+-------------+---------+
id is the primary key (column with unique values) for this table.
Each row of this table indicates the name of an employee, their department, and the id of their manager.
If managerId is null, then the employee does not have a manager.
No employee will be the manager of themself.

Write a solution to find managers with at least **five direct reports**.

Return the result table in **any order**.

```
Easiest solution: 
```
select name
from Employee
where id in
(select managerID
from employee
group by managerId
having count(*)>=5)
```
## Problem 5
```
Table: `Signups`

+----------------+----------+
| Column Name    | Type     |
+----------------+----------+
| user_id        | int      |
| time_stamp     | datetime |
+----------------+----------+
user_id is the column of unique values for this table.
Each row contains information about the signup time for the user with ID user_id.

Table: `Confirmations`

+----------------+----------+
| Column Name    | Type     |
+----------------+----------+
| user_id        | int      |
| time_stamp     | datetime |
| action         | ENUM     |
+----------------+----------+
(user_id, time_stamp) is the primary key (combination of columns with unique values) for this table.
user_id is a foreign key (reference column) to the Signups table.
action is an ENUM (category) of the type ('confirmed', 'timeout')
Each row of this table indicates that the user with ID user_id requested a confirmation message at time_stamp and that confirmation message was either confirmed ('confirmed') or expired without confirming ('timeout').

The **confirmation rate** of a user is the number of `'confirmed'` messages divided by the total number of requested confirmation messages. The confirmation rate of a user that did not request any confirmation messages is `0`. Round the confirmation rate to **two decimal** places.

Write a solution to find the **confirmation rate** of each user.

Return the result table in **any order**.
```
Solution: 
```
SELECT a.user_id, round(avg(case when action = 'confirmed' then 1 else 0 end),2)
as confirmation_rate
FROM Signups a
LEFT JOIN Confirmations b
ON a.user_id = b.user_id
GROUP BY a.user_id
```
## Problem 6
```
+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| product_id    | int     |
| start_date    | date    |
| end_date      | date    |
| price         | int     |
+---------------+---------+
(product_id, start_date, end_date) is the primary key (combination of columns with unique values) for this table.
Each row of this table indicates the price of the product_id in the period from start_date to end_date.
For each product_id there will be no two overlapping periods. That means there will be no two intersecting periods for the same product_id.

Table: `UnitsSold`

+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| product_id    | int     |
| purchase_date | date    |
| units         | int     |
+---------------+---------+
This table may contain duplicate rows.
Each row of this table indicates the date, units, and product_id of each product sold. 

Write a solution to find the average selling price for each product. `average_price` should be **rounded to 2 decimal places**.
```

MySQL solution:
```
SELECT p.product_id, IFNULL(ROUND(SUM(units*price)/SUM(units),2),0) AS average_price
FROM Prices p LEFT JOIN UnitsSold u
ON p.product_id = u.product_id AND
u.purchase_date BETWEEN start_date AND end_date
group by product_id
```

## Problem 7

## SQL tutorials

### INSERT INTO
The `INSERT INTO` statement is used to insert new records in a table.
```
INSERT INTO _table_name_ (_column1_, _column2_, _column3_, ...)  
VALUES (_value1_, _value2_, _value3_, ...);
```
### UPDATE and DELETE
```
UPDATE _table_name_  
SET _column1_ = _value1_, _column2_ = _value2_, ...  
WHERE _condition_;
```

```
DELETE FROM _table_name_ WHERE _condition_;
```
###  TOP, LIMIT, FETCH FIRST or ROWNUM Clause

- The `SELECT TOP` clause is used to specify the number of records to return
> **Note:** Not all database systems support the `SELECT TOP` clause. MySQL supports the `LIMIT` clause to select a limited number of records, while Oracle uses `FETCH FIRST _n_ ROWS ONLY` and `ROWNUM`

### Aggregate functions

The most commonly used SQL aggregate functions are:

- `MIN()` - returns the smallest value within the selected column
- `MAX()` - returns the largest value within the selected column
- `COUNT()` - returns the number of rows in a set
- `SUM()` - returns the total sum of a numerical column
- `AVG()` - returns the average value of a numerical column

### LIKE operator

The `LIKE` operator is used in a `WHERE` clause to search for a specified pattern in a column.

There are two wildcards often used in conjunction with the `LIKE` operator:

-  The percent sign `%` represents zero, one, or multiple characters
-  The underscore sign `_` represents one, single character

Select all customers that starts with the letter "a":
```
SELECT * FROM Customers  
WHERE CustomerName LIKE 'a%';
```

|Symbol|Description|
|---|---|
|%|Represents zero or more characters|
|_|Represents a single character|
|[]|Represents any single character within the brackets *|
|^|Represents any character not in the brackets *|
|-|Represents any single character within the specified range *|
|{}|Represents any escaped character **|

* Not supported in PostgreSQL and MySQL databases.

** Supported only in Oracle databases.

### UNION operator
```
SELECT _column_name(s)_ FROM _table1_  
UNION  
SELECT _column_name(s)_ FROM _table2_;
```

The `UNION` operator selects only distinct values by default. To allow duplicate values, use `UNION ALL`.

### GROUP BY and HAVING
```
SELECT _column_name(s)_  
FROM _table_name_  
WHERE _condition_  
GROUP BY _column_name(s)  
_ORDER BY _column_name(s);_
```

`HAVING` keyword was added to SQL since `WHERE` can't handle aggregate functions.
```
SELECT _column_name(s)_  
FROM _table_name_  
WHERE _condition_  
GROUP BY _column_name(s)  
_HAVING _condition  
_ORDER BY _column_name(s);_
```
### EXISTS operator
```
SELECT _column_name(s)_  
FROM _table_name_  
WHERE EXISTS  
(SELECT _column_name_ FROM _table_name_ WHERE _condition_);
```
### ANY and ALL
The `ANY` operator:

- returns a boolean value as a result
- returns TRUE if ANY of the subquery values meet the condition

The `ALL` operator:

- returns a boolean value as a result
- returns TRUE if ALL of the subquery values meet the condition
- is used with `SELECT`, `WHERE` and `HAVING` statements

### CASE 
```
CASE  
    WHEN _condition1_ THEN _result1_  
    WHEN _condition2_ THEN _result2_  
    WHEN _conditionN_ THEN _resultN_  
    ELSE _result_  
END;
```
### NULL Functions

**MySQL**

The MySQL `[IFNULL()]` function lets you return an alternative value if an expression is NULL:

`SELECT ProductName, UnitPrice * (UnitsInStock + IFNULL(UnitsOnOrder, 0))  FROM Products;`

or we can use the `[COALESCE()]` function, like this:

`SELECT ProductName, UnitPrice * (UnitsInStock + COALESCE(UnitsOnOrder, 0))  FROM Products;`

### Stored procedures
A stored procedure is a prepared SQL code that you can save, so the code can be reused over and over again.

So if you have an SQL query that you write over and over again, save it as a stored procedure, and then just call it to execute it.

```
CREATE PROCEDURE _procedure_name_  
AS  
_sql_statement_  
GO;

EXEC _procedure_name_; -> to execute stored procedure
```

You can also add multiple parameters to a procedure:
```
CREATE PROCEDURE SelectAllCustomers @City nvarchar(30), @PostalCode nvarchar(10)  
AS  
SELECT * FROM Customers WHERE City = @City AND PostalCode = @PostalCode  
GO;

EXEC SelectAllCustomers @City = 'London', @PostalCode = 'WA1 1DP';
```

### CREATE DATABASE

`CREATE DATABASE _databasename_;` to create new database

`DROP DATABASE _databasename_;` -> to drop an existing database

```
BACKUP DATABASE _databasename_  
TO DISK = '_filepath_'; -> for database backup
```
OR 
```
BACKUP DATABASE _databasename_  
TO DISK = '_filepath_'  
WITH DIFFERENTIAL;
```
### CREATE TABLE
```
CREATE TABLE _table_name_ (  
    _column1 datatype_,  
    _column2 datatype_,  
    _column3 datatype_,  
   ....  
);
```

### ALTER TABLE

```
ALTER TABLE _table_name_  
ADD _column_name datatype_;
```

```
ALTER TABLE _table_name_  
DROP COLUMN _column_name_;
```

```
ALTER TABLE _table_name_  
RENAME COLUMN _old_name_ to _new_name_;
```

```
ALTER TABLE _table_name_  
ALTER COLUMN _column_name datatype_;
```

### SQL Constraints
```
SQL constraints are used to specify rules for the data in a table.

Constraints are used to limit the type of data that can go into a table. This ensures the accuracy and reliability of the data in the table. If there is any violation between the constraint and the data action, the action is aborted.

Constraints can be column level or table level. Column level constraints apply to a column, and table level constraints apply to the whole table.

The following constraints are commonly used in SQL:

- `[NOT NULL] - Ensures that a column cannot have a NULL value
- `[UNIQUE]- Ensures that all values in a column are different
- `[PRIMARY KEY] - A combination of a `NOT NULL` and `UNIQUE`. Uniquely identifies each row in a table
- `[FOREIGN KEY] - Prevents actions that would destroy links between tables
- `[CHECK] - Ensures that the values in a column satisfies a specific condition
- `[DEFAULT] - Sets a default value for a column if no value is specified
- `[CREATE INDEX] - Used to create and retrieve data from the database very quickly
```
### SQL AUTO INCREMENT Field
```
CREATE TABLE Persons (  
    Personid int NOT NULL AUTO_INCREMENT,  
    LastName varchar(255) NOT NULL,  
    FirstName varchar(255),  
    Age int,  
    PRIMARY KEY (Personid)  
);
```
Auto-increment allows a unique number to be generated automatically when a new record is inserted into a table.

# Key-value databases

Examples: Redis,Memcached

**Used/good for**: 
- Good for unstructured data
-  Uses hash-tabels for fast lookups and storage
-  Often stored in memory , which makes retrieval fast and scales with size
- **Horizontal Scaling**: Key-value databases offer the advantage of horizontal scaling, allowing applications to scale out by adding more servers to distribute data and workload across multiple nodes. This scalability feature ensures that performance levels are maintained even as data volumes grow, making them suitable for applications with increasing user demands and massive data storage needs
- Best for caching

**Not suitable for**:
- Not for complex data models/quaries
- Not for transactional consistency : Transactional consistency refers to the state of a database where a transaction, when executed in isolation without interference from other concurrent transactions, preserves the consistency of the database. 
- Not for historic-data, no data compression or indexing
# Wide-columns databases

Examples:  Cassandra, Apache Hbase

**Used/good for**: 
- Highly partitionable/ Horizontal / Queries by primary key
-  Denormalized -> faster data retrieval
- Light-weight transactions
- Great for scalability/Horizontal scaling
- Best for writes

**Not suitable for**:
- Not for filtering, joins or aggregation
-  Data inconsistencies due to denormalized nature and data duplicates 
- Poor transactions
# Document databases

Examples: MongoDB, DynamoDB, CosmosDB

**Used/good for**: 
- Denormalized
- Good for unstructured data/Flexible schema
- Rich queries (compound indexing, multikey indexing, full-text indexing, hash indexing, single field indexing)
- Highly scalable
- Perfect match with OOP
- Decent for transactions

 **Not suitable for**:
- Inconsistencies/duplicated data
- Not for complex relationships/ joins
- Poor referential integrity

# Relational databases

Examples: MySQL,PostgreSQL

**Used/good for**: 
- Interconnected data
- Normalized/ Data integrity
- ACID (Atomicity Consistency Isolation Durability)

 **Not suitable for**:
 - Scaling difficulties
 - Extremely large traffic requirements

# Graph databases

Examples: ArangoDB

**Used/good for**: 
- No need to compute the relationships
- Best for multi-hop relationships

 **Not suitable for**:
 - Managing complexities
 - Scalability constraints
 -  Not for write heavy works / high risk of data inconsistency



# Database normalization

**Normalized tables are**:
- Easier to understand 
- Easier to enhance and extend
- Protected from
	 insertion anomalies
	 update anomalies
	 deletion anomalies
## 1NF

- Using row order to convey information voilates 1NF (Use specific columns for this)
- Mixing datatype violates 1NF
- Designing a table without a primary key violates 1NF (ADD PRIMARY KEY {column})
- Storing a repeating group of data items on a single row violates 1NF
## 2NF 

-  Each non-key attribute must depend on the entire primary key
## 3NF

- Every non-key attribute in a table should depend on the key, the whole key and nothing but the key

OR **Boyce-Codd normal form**:
Every  attribute in a table should depend on the key, the whole key and nothing but the key
## 4NF

- Multivalued dependencies in a table must be multivalued dependencies on the key
## 5NF
- the table (which must be in 4NF) cannot be describable as the logical result of joining some other tables

 



# Database keys

- Primary key
	- No null allowed
	- Binds the table
- Сandidate key
	- Must exhibit uniqueness across time
-  Superkey
	- Consists of candidate key's attributes + potentially some extra attributes
-  Alternate key
	-  Used to ensure uniqueness of second candidate key
- Foreign keys
	-  Ensure uniqueness
- Surrogate and Natural
	-  A surrogate key is a system-generated value with no business meaning that is used to uniquely identify a record in a table. It is typically an auto-numbered or auto-incremented value, a GUID, or a sequence that serves as a unique identifier for a row.
	- A natural key is a column or set of columns that already exist in the table and uniquely identify a record based on their business meaning. These keys are attributes of the entity within the data model and have inherent business significance.
- Compound keys
	- A compound key is a candidate key formed by combining multiple attributes that, when taken together, uniquely identify a record in a table.
- Intelligent keys
 