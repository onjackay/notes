# Curiously Recurring Template Pattern (CRTP)

CRTP：基类模板以派生类作为模板参数，从而在编译期实现静态多态（static polymorphism）。

### 实现静态多态

CRTP 通过在基类公开接口，并在派生类中实现该接口，从而在编译期绑定函数调用，实现静态多态，避免虚函数的运行时开销。

```cpp
template<typename Derived>
class Shape {
public:
    float area() const {
        return static_cast<const Derived*>(this)->area();
    }
};

class Circle : public Shape<Circle> {
public:
    Circle(float radius) : radius(radius) {}
    float area() const {
        return 3.14159f * radius * radius;
    }
private:
    float radius;
};

class Square : public Shape<Square> {
public:
    Square(float side) : side(side) {}
    float area() const {
        return side * side;
    }
private:
    float side;
};

// 使用：
Circle c(5.0f);
Square s(4.0f);
float circleArea = c.area(); // 78.5398
float squareArea = s.area(); // 16.0
```

### 自定义派生类的行为

CRTP 可以给基类提供通用接口，派生类实现具体行为。可以把派生类的公共逻辑放在基类中，派生类只需实现特定行为，减少代码重复。

```cpp
template <typename Derived>
class EqualityComparable {
public:
    bool operator==(const Derived& other) const {
        const Derived& self = static_cast<const Derived&>(*this);
        return self.is_equal(other);
    }

    bool operator!=(const Derived& other) const {
        return !(*this == other);
    }
};

class MyClass : public EqualityComparable<MyClass> {
    int value;
public:
    MyClass(int v) : value(v) {}

    bool is_equal(const MyClass& other) const {
        return value == other.value;
    }
};

// 使用：
MyClass a(1), b(2);
if (a == b) { ... } // 自动可用！
```

3. 实现编译期静态接口约束，类似 C++20 concepts，用于要求派生类必须实现某个函数

```cpp
template <typename Derived>
class Drawable {
public:
    void draw() {
        static_cast<Derived*>(this)->do_draw();
    }
};
// 如果 Derived 没有 do_draw()，编译时报错
```

在 C++20 后的版本中，可以使用 concepts 来实现类似的功能：

```cpp
template <typename T>
concept DrawableConcept = requires(T t) {
    { t.do_draw() };
};

template <DrawableConcept T>
class Drawable {
public:
    void draw() {
        static_cast<T*>(this)->do_draw();
    }
};
```

### 注意事项

- CRTP 需要派生类在编译期已知，因此不能用于运行时多态：不能使用指针或引用来存储基类类型。

### 用例

`std::enable_shared_from_this` 是 CRTP 的一个常见用例，它允许类安全地从 `shared_from_this` 获取自身的 `shared_ptr`。

```cpp
class Widget: public std::enable_shared_from_this<Widget> {
public:
…
void process();
…
}
```

简化版实现：

```cpp
template <typename T>
class enable_shared_from_this {
protected:
    weak_ptr<T> __weak_this; // 内部保存一个弱引用，指向管理 this 的 shared_ptr

public:
    shared_ptr<T> shared_from_this() {
        return shared_ptr<T>(__weak_this); // 从弱引用升级为 shared_ptr
    }
};
```

当对象被第一个 `shared_ptr` 构造时，标准库会检测它是否继承自`enable_shared_from_this<T>`，如果是，就自动设置 `__weak_this` 指向这个控制块。
