// Minimal JSON reader for the configs stored in safetensors metadata (objects, arrays, numbers, strings, bools).
#pragma once

#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rvc::detail {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    static Json parse(const std::string& text) {
        size_t pos = 0;
        Json value = parseValue(text, pos);
        skipSpace(text, pos);
        if (pos != text.size()) throw std::runtime_error("JSON: trailing characters");
        return value;
    }

    Type type() const { return type_; }
    bool has(const std::string& key) const { return type_ == Type::Object && object_.count(key) > 0; }
    const Json& operator[](const std::string& key) const {
        auto it = object_.find(key);
        if (type_ != Type::Object || it == object_.end()) throw std::runtime_error("JSON: missing key '" + key + "'");
        return it->second;
    }
    const Json& operator[](size_t i) const {
        if (type_ != Type::Array || i >= array_.size()) throw std::runtime_error("JSON: bad array index");
        return array_[i];
    }
    size_t size() const { return array_.size(); }
    double number() const { expect(Type::Number); return number_; }
    int integer() const { return static_cast<int>(number()); }
    bool boolean() const { expect(Type::Bool); return bool_; }
    const std::string& string() const { expect(Type::String); return string_; }
    std::vector<int> ints() const {
        std::vector<int> out;
        for (const auto& v : array_) out.push_back(v.integer());
        return out;
    }

private:
    void expect(Type t) const {
        if (type_ != t) throw std::runtime_error("JSON: unexpected value type");
    }
    static void skipSpace(const std::string& s, size_t& p) {
        while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
    }
    static Json parseValue(const std::string& s, size_t& p) {
        skipSpace(s, p);
        if (p >= s.size()) throw std::runtime_error("JSON: unexpected end");
        Json v;
        char c = s[p];
        if (c == '{') {
            v.type_ = Type::Object;
            ++p;
            skipSpace(s, p);
            if (s[p] == '}') { ++p; return v; }
            while (true) {
                skipSpace(s, p);
                std::string key = parseString(s, p);
                skipSpace(s, p);
                if (s[p++] != ':') throw std::runtime_error("JSON: expected ':'");
                v.object_[key] = parseValue(s, p);
                skipSpace(s, p);
                if (s[p] == ',') { ++p; continue; }
                if (s[p] == '}') { ++p; return v; }
                throw std::runtime_error("JSON: expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.type_ = Type::Array;
            ++p;
            skipSpace(s, p);
            if (s[p] == ']') { ++p; return v; }
            while (true) {
                v.array_.push_back(parseValue(s, p));
                skipSpace(s, p);
                if (s[p] == ',') { ++p; continue; }
                if (s[p] == ']') { ++p; return v; }
                throw std::runtime_error("JSON: expected ',' or ']'");
            }
        }
        if (c == '"') { v.type_ = Type::String; v.string_ = parseString(s, p); return v; }
        if (s.compare(p, 4, "true") == 0) { v.type_ = Type::Bool; v.bool_ = true; p += 4; return v; }
        if (s.compare(p, 5, "false") == 0) { v.type_ = Type::Bool; v.bool_ = false; p += 5; return v; }
        if (s.compare(p, 4, "null") == 0) { p += 4; return v; }
        char* end = nullptr;
        v.number_ = std::strtod(s.c_str() + p, &end);
        if (end == s.c_str() + p) throw std::runtime_error("JSON: bad value");
        v.type_ = Type::Number;
        p = static_cast<size_t>(end - s.c_str());
        return v;
    }
    static std::string parseString(const std::string& s, size_t& p) {
        if (s[p] != '"') throw std::runtime_error("JSON: expected string");
        std::string out;
        for (++p; p < s.size() && s[p] != '"'; ++p) {
            if (s[p] == '\\' && p + 1 < s.size()) {
                char e = s[++p];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'u': out += '?'; p += 4; break;  // configs are ASCII; non-ASCII names are only displayed
                    default: out += e;
                }
            } else {
                out += s[p];
            }
        }
        ++p;
        return out;
    }

    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0;
    std::string string_;
    std::vector<Json> array_;
    std::map<std::string, Json> object_;
};

}  // namespace rvc::detail
