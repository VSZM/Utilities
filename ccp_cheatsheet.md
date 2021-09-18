# **C++ Cheatsheet**

## **String methods**

#### String split by char

This will split the original string by a given delimiter and put it into a vector. It is exploiting an overload of getline function, where we pass the delimiter.
Note that this only works for a single character delimiter.

```c++
vector<string> strings;
istringstream f("denmark;sweden;india;us");
string s;    
while (getline(f, s, ';')) {
    cout << s << endl;
    strings.push_back(s);
}
```


#### String to number

```
int std::stoi(string)
long std::stol(string)
float stof(string)
double stod(string)
```
