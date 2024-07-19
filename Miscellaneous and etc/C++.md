# Loops
```
for (int i = 0; i <= 5 , ++i) {statement}
```

# Type modifications

1. `(type)expression` -> worst
2. `static_cast<int>(exp)` -> safest. Check at compile rime, but can result in undefined behaviour during runtime
3. `dynamic_cast` -> generally not recommended. Checks at runtime for conversion between related classes, but only with virtual functions, has higher runtime overhead.
4. `reinterpt_cast` -> low-level conversion of any datatype, raw memory, undefined behaviour as cause of no checks
5. `const_cast`

# Containers
## include array
Classic array implementation
```
#inlcude <array>
#include <algorithm>
std::array<type,size> b;

std::fill(b.begin(),b.end(),something()); -> safest way
OR
for (int i=0;i<b.size();++i) {} 
OR 
for (std::array<int,10>::iterator i = b.begin();i != b.end(); ++i) {} -> Nah, not this shit.

for (auto i = b.begin();i != b.end(); ++i) {} -> Nah, not this shit.

for (auto& i : b) {}
```
## std::vector

One problem -> If the memory is exceeded ,the vector will copy itself into a free space, leading to a memory leak (and pointers fail)
```
#include <vector>

std::vector<int> b;
```  
## std::list
```
#include <list>
std::list<int> b; -> linked list basically
```
## std::deque 
```
#include <deque>
std::deque<int> b;
```
## std::set and unordered_set
```
#incldue <set>
std::set<int> container;
insert() instead of push_back() method
```
Then there's unordered_set
```
#incldue <unordered_set>
std::unordered_set<int>
```
## map and unordered map
```
std::unordered_map<std::string ,int> container;
container["one"]=1;

```
# Pointers and references
```
int* x;
*x;
int* y = &b;
int& y = x;
сonst int* p = x; -> can change p, but can't change the *p
Or int* const p = x; can change x, but can't change p
```
# Functions
Normal function:
```
type name(type x,type y) {...}
```
Lambda/Function pointer:
```
type (*p)(type x, type y) = &f
```
Void function is simply:
```
void f(int x) {...}
```
# Declarations and classes

## Declarations
Free up memory space:
```
new type(defaul)
int* p = new int;
delete[] p if p is an array or simply delete p
```
## Classes
```
#include <iostream> 
#include <cstring> 
class String 
{ private: 
	char* data; 
	size_t length; 
  public: // 
	  String() : data(nullptr), length(0) {} 
	  String(const char* str) { 
		  length = strlen(str); 
		  data = new char[length + 1]; 
		  strcpy(data, str); 
		  } 
	   String(const String& other) {     //copy constructor
		   length = other.length; 
		   data = new char[length + 1]; 
		   strcpy(data, other.data); 
		   } 
	    ~String() {    //destructor
		    delete[] data; 
		    } 
		String& operator=(const String& other) {   //Defining operator
			if (this != &other) 
			{ delete[] data; 
			length = other.length; 
			data = new char[length + 1]; 
			strcpy(data, other.data); 
			} 
			return *this; } 
		char& operator[](size_t index) { 
			return data[index]; 
		    } 
		const char& operator[](size_t index) const { 
			return data[index]; 
			} 
		friend std::ostream& operator<<(std::ostream& os, const String& str); 
		friend std::istream& operator>>(std::istream& is, String& str);  
		size_t size() const { return length; } 
		const char* c_str() const { return data; } 
		void append(const String& other) { 
			size_t newLength = length + other.length; 
			char* newData = new char[newLength + 1]; 
			strcpy(newData, data); 
			strcat(newData, other.data); 
			delete[] data; 
			data = newData; 
			length = newLength; }
}; 
```
 

# Overloading operators
```
class& operator +(const class& x) {};
```