#include <iostream>

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

int main() {
    Circle circle(5.0f);
    Square square(4.0f);

    std::cout << "Circle area: " << circle.area() << std::endl;
    std::cout << "Square area: " << square.area() << std::endl;

    return 0;
}