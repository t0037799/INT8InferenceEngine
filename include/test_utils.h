#include <iostream>

template <typename T>
void print(T arg) {
  std::cerr << arg << "\n";
}
template <>
void print<char>(char arg) {
  std::cerr << (int)arg << "\n";
}
template <>
void print<unsigned char>(unsigned char arg) {
  std::cerr << (int)arg << "\n";
}
template <typename T>
void p(T arg) {
  std::cerr << arg << " ";
}
template <>
void p<unsigned char>(unsigned char arg) {
  std::cerr << (int)arg << " ";
}
template <typename T, typename... ARGS>
void print(T arg, ARGS... args) {
  p(arg);
  print(args...);
}
