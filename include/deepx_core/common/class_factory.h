// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/dx_log.h>
#include <set>
#include <string>
#include <type_traits>  // std::enable_if, ...
#include <unordered_map>

namespace deepx_core {

/************************************************************************/
/* ClassFactory */
/************************************************************************/
template <typename T> // T 是基类
class ClassFactory {
 public:
  using value_type = T;
  using pointer = value_type*;
  using creator = pointer (*)();

 private:
  std::unordered_map<std::string, creator> map_;
  std::set<std::string> set_;

 public:
  void Register(const std::string& name, creator creator) { // name 是子类的类名，creator 是创建子类的
    if (map_.count(name) > 0) {
      DXTHROW_INVALID_ARGUMENT("Duplicate registered name: %s.", name.c_str());
    }
    map_.emplace(name, creator);
    set_.emplace(name);
  }

  pointer New(const std::string& name) const {
    auto it = map_.find(name);
    if (it == map_.end()) {
      DXERROR("Unregistered name: %s.", name.c_str());
      return nullptr;
    }
    return it->second(); // 调用 creator 函数创建子类
  }

  const std::set<std::string>& Names() const noexcept { return set_; } // 所有 T 的子类的名称

 private:
  ClassFactory() = default;

 public:
  static ClassFactory& GetInstance() {
    static ClassFactory factory;
    return factory;
  }
};

/************************************************************************/
/* ClassFactoryRegister */
/************************************************************************/
template <typename T, typename U, // 通过定义一个 ClassFactoryRegister 对象来注册一个类到类工厂，同时指定基类 T 和 子类 U
          typename =
              typename std::enable_if<std::is_default_constructible<U>::value &&
                                      std::is_convertible<U*, T*>::value>::type>
class ClassFactoryRegister {
 private:
  static T* Create() { return new U; } // 统一的创建子类对象的方式，所以 U 必须是 is_default_contructible() 的

 public:
  explicit ClassFactoryRegister(const std::string& name) { // name 是子类类名或者标识，仅做 name -> creator 的寻址
    ClassFactory<T>::GetInstance().Register(name, &Create);
  }
};

/************************************************************************/
/* Helper macros */
/************************************************************************/ // T基类，U子类，name子类的标识或者名称（不一定是类名）
#define _CLASS_FACTORY_CONCAT_IMPL(x, y, z) _##x##_##y##_##z
#define _CLASS_FACTORY_CONCAT(x, y, z) _CLASS_FACTORY_CONCAT_IMPL(x, y, z)
#define CLASS_FACTORY_REGISTER(T, U, name)                      \
  deepx_core::ClassFactoryRegister<T, U> _CLASS_FACTORY_CONCAT( \
      T, U, __COUNTER__)(name)

#define CLASS_FACTORY_NEW(T, name) \
  deepx_core::ClassFactory<T>::GetInstance().New(name)

#define CLASS_FACTORY_NAMES(T) \
  deepx_core::ClassFactory<T>::GetInstance().Names()

}  // namespace deepx_core
