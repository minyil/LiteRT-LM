// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "omni/tts/text_normalizer.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::omni::tts {
namespace {

// Highest capture group a replacement may reference. RE2 supports far more,
// but a rule needing ten groups is a sign the table wants a second rule.
constexpr int kMaxGroup = 9;

// Returns the length in bytes of the UTF-8 character starting at `text[0]`,
// or 1 if the byte is not a well-formed leading byte. Used only to guarantee
// forward progress past an empty match, so a wrong answer costs nothing worse
// than splitting one malformed character.
size_t Utf8CharLength(absl::string_view text) {
  if (text.empty()) return 0;
  const unsigned char lead = static_cast<unsigned char>(text[0]);
  if (lead < 0x80) return 1;
  if ((lead & 0xE0) == 0xC0) return 2;
  if ((lead & 0xF0) == 0xE0) return 3;
  if ((lead & 0xF8) == 0xF0) return 4;
  return 1;
}

// Reports whether `replacement` uses a `#N` digit expansion, which is what
// makes the digit glyph table mandatory.
bool UsesDigitExpansion(absl::string_view replacement) {
  for (size_t i = 0; i + 1 < replacement.size(); ++i) {
    if (replacement[i] == '\\') {
      ++i;  // Skip the escaped character.
      continue;
    }
    if (replacement[i] == '#' && absl::ascii_isdigit(replacement[i + 1])) {
      return true;
    }
  }
  return false;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TextNormalizer>> TextNormalizer::Create(
    absl::string_view rule_table) {
  auto normalizer = std::unique_ptr<TextNormalizer>(new TextNormalizer());
  bool needs_digits = false;
  int digits_defined = 0;

  int line_number = 0;
  for (absl::string_view line : absl::StrSplit(rule_table, '\n')) {
    ++line_number;
    // Tolerate CRLF so a table edited on Windows still parses.
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    const absl::string_view trimmed = absl::StripLeadingAsciiWhitespace(line);
    if (trimmed.empty() || trimmed.front() == '#') continue;

    const std::vector<absl::string_view> fields = absl::StrSplit(line, '\t');
    if (fields[0] == "digit") {
      if (fields.size() != 3) {
        return absl::InvalidArgumentError(absl::StrCat(
            "text normalizer rule table line ", line_number,
            ": `digit` takes 2 tab-separated fields, got ", fields.size() - 1));
      }
      if (fields[1].size() != 1 || !absl::ascii_isdigit(fields[1][0])) {
        return absl::InvalidArgumentError(absl::StrCat(
            "text normalizer rule table line ", line_number,
            ": `digit` expects a single ASCII digit, got \"", fields[1], "\""));
      }
      if (fields[2].empty()) {
        return absl::InvalidArgumentError(
            absl::StrCat("text normalizer rule table line ", line_number,
                         ": `digit` glyph is empty"));
      }
      const int value = fields[1][0] - '0';
      if (!normalizer->digit_glyphs_[value].empty()) {
        return absl::InvalidArgumentError(
            absl::StrCat("text normalizer rule table line ", line_number,
                         ": digit ", value, " is defined twice"));
      }
      normalizer->digit_glyphs_[value] = std::string(fields[2]);
      ++digits_defined;
    } else if (fields[0] == "rule") {
      if (fields.size() != 3) {
        return absl::InvalidArgumentError(absl::StrCat(
            "text normalizer rule table line ", line_number,
            ": `rule` takes 2 tab-separated fields, got ", fields.size() - 1));
      }
      RE2::Options options;
      options.set_log_errors(false);
      auto pattern = std::make_unique<RE2>(fields[1], options);
      if (!pattern->ok()) {
        return absl::InvalidArgumentError(
            absl::StrCat("text normalizer rule table line ", line_number,
                         ": cannot compile pattern \"", fields[1],
                         "\": ", pattern->error()));
      }
      if (pattern->NumberOfCapturingGroups() > kMaxGroup) {
        return absl::InvalidArgumentError(absl::StrCat(
            "text normalizer rule table line ", line_number, ": pattern has ",
            pattern->NumberOfCapturingGroups(), " capture groups, at most ",
            kMaxGroup, " are addressable"));
      }
      if (UsesDigitExpansion(fields[2])) needs_digits = true;
      normalizer->rules_.push_back(
          Rule{std::move(pattern), std::string(fields[2])});
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("text normalizer rule table line ", line_number,
                       ": unknown record type \"", fields[0],
                       "\", expected `digit` or `rule`"));
    }
  }

  if (needs_digits && digits_defined != 10) {
    return absl::InvalidArgumentError(absl::StrCat(
        "text normalizer rule table uses a `#N` digit expansion but defines ",
        digits_defined, " of the 10 digits"));
  }
  return normalizer;
}

size_t TextNormalizer::RuleCount() const { return rules_.size(); }

void TextNormalizer::ExpandReplacement(absl::string_view replacement,
                                       absl::string_view* groups,
                                       int group_count,
                                       std::string& out) const {
  for (size_t i = 0; i < replacement.size(); ++i) {
    const char c = replacement[i];
    const bool has_next = i + 1 < replacement.size();
    if (c == '\\' && has_next) {
      const char next = replacement[i + 1];
      ++i;
      switch (next) {
        case 't':
          out.push_back('\t');
          break;
        case 'n':
          out.push_back('\n');
          break;
        default:
          if (absl::ascii_isdigit(next)) {
            const int group = next - '0';
            if (group < group_count) out.append(groups[group]);
          } else {
            // Covers \\ and \#, and leaves any other escape as the bare
            // character so a table author cannot accidentally lose content.
            out.push_back(next);
          }
          break;
      }
      continue;
    }
    if (c == '#' && has_next && absl::ascii_isdigit(replacement[i + 1])) {
      const int group = replacement[i + 1] - '0';
      ++i;
      if (group >= group_count) continue;
      for (const char digit_char : groups[group]) {
        if (absl::ascii_isdigit(digit_char)) {
          out.append(digit_glyphs_[digit_char - '0']);
        } else {
          out.push_back(digit_char);
        }
      }
      continue;
    }
    out.push_back(c);
  }
}

std::string TextNormalizer::ApplyRule(const Rule& rule,
                                      absl::string_view text) const {
  const int group_count = rule.pattern->NumberOfCapturingGroups() + 1;
  std::array<absl::string_view, kMaxGroup + 1> groups;

  std::string out;
  size_t pos = 0;
  while (pos <= text.size()) {
    if (!rule.pattern->Match(text, pos, text.size(), RE2::UNANCHORED,
                             groups.data(), group_count)) {
      break;
    }
    const size_t match_begin = groups[0].data() - text.data();
    const size_t match_end = match_begin + groups[0].size();
    out.append(text.substr(pos, match_begin - pos));
    ExpandReplacement(rule.replacement, groups.data(), group_count, out);
    if (match_end == match_begin) {
      // An empty match would spin forever. Emit one character and move on.
      const size_t step = Utf8CharLength(text.substr(match_end));
      if (step == 0) return absl::StrCat(out, text.substr(match_end));
      out.append(text.substr(match_end, step));
      pos = match_end + step;
      continue;
    }
    pos = match_end;
  }
  out.append(text.substr(pos));
  return out;
}

std::string TextNormalizer::Normalize(absl::string_view text) const {
  std::string current(text);
  for (const Rule& rule : rules_) {
    current = ApplyRule(rule, current);
  }
  return current;
}

}  // namespace litert::omni::tts
