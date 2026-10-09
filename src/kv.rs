//! Valve KeyValues text format, the syntax `.p2c` files are written in.
//!
//! Order and duplicate keys are preserved, and quoted strings are kept raw (escapes are not
//! interpreted), so `write(parse(s)) == s` for files in the game's own formatting.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Str(String),
    Block(Vec<Pair>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pair {
    pub key: String,
    pub value: Value,
}

impl Pair {
    pub fn str(key: impl Into<String>, value: impl Into<String>) -> Pair {
        Pair {
            key: key.into(),
            value: Value::Str(value.into()),
        }
    }

    pub fn block(key: impl Into<String>, children: Vec<Pair>) -> Pair {
        Pair {
            key: key.into(),
            value: Value::Block(children),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.msg)
    }
}

impl std::error::Error for ParseError {}

enum Tok {
    Str(String),
    Open,
    Close,
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
    line: usize,
}

impl<'a> Lexer<'a> {
    fn err(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            line: self.line,
            msg: msg.into(),
        }
    }

    fn next(&mut self) -> Result<Option<Tok>, ParseError> {
        let bytes = self.src.as_bytes();
        loop {
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                if bytes[self.pos] == b'\n' {
                    self.line += 1;
                }
                self.pos += 1;
            }
            if self.src[self.pos..].starts_with("//") {
                while self.pos < bytes.len() && bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            break;
        }
        let Some(&b) = bytes.get(self.pos) else {
            return Ok(None);
        };
        match b {
            b'{' => {
                self.pos += 1;
                Ok(Some(Tok::Open))
            }
            b'}' => {
                self.pos += 1;
                Ok(Some(Tok::Close))
            }
            b'"' => {
                let start = self.pos + 1;
                let mut i = start;
                while i < bytes.len() && bytes[i] != b'"' {
                    match bytes[i] {
                        b'\\' => i += 1,
                        b'\n' => self.line += 1,
                        _ => {}
                    }
                    i += 1;
                }
                if i >= bytes.len() {
                    return Err(self.err("unterminated string"));
                }
                self.pos = i + 1;
                Ok(Some(Tok::Str(self.src[start..i].to_string())))
            }
            _ => {
                let start = self.pos;
                while self.pos < bytes.len()
                    && !bytes[self.pos].is_ascii_whitespace()
                    && !matches!(bytes[self.pos], b'{' | b'}' | b'"')
                {
                    self.pos += 1;
                }
                Ok(Some(Tok::Str(self.src[start..self.pos].to_string())))
            }
        }
    }
}

pub fn parse(src: &str) -> Result<Vec<Pair>, ParseError> {
    let mut lex = Lexer {
        src,
        pos: 0,
        line: 1,
    };
    parse_pairs(&mut lex, true)
}

fn parse_pairs(lex: &mut Lexer, top: bool) -> Result<Vec<Pair>, ParseError> {
    let mut pairs = Vec::new();
    loop {
        let key = match lex.next()? {
            None if top => return Ok(pairs),
            None => return Err(lex.err("unexpected end of file inside block")),
            Some(Tok::Close) if !top => return Ok(pairs),
            Some(Tok::Close) => return Err(lex.err("unmatched '}'")),
            Some(Tok::Open) => return Err(lex.err("unexpected '{' where a key was expected")),
            Some(Tok::Str(k)) => k,
        };
        let value = match lex.next()? {
            Some(Tok::Str(v)) => Value::Str(v),
            Some(Tok::Open) => Value::Block(parse_pairs(lex, false)?),
            _ => return Err(lex.err(format!("key \"{key}\" has no value"))),
        };
        pairs.push(Pair { key, value });
    }
}

/// Writes pairs in the Puzzle Maker's formatting: tab indents, `"key"<tab><tab>"value"`, LF.
pub fn write(pairs: &[Pair]) -> String {
    let mut out = String::new();
    write_into(&mut out, pairs, 0);
    out
}

fn write_into(out: &mut String, pairs: &[Pair], depth: usize) {
    for pair in pairs {
        indent(out, depth);
        match &pair.value {
            Value::Str(v) => {
                out.push('"');
                out.push_str(&pair.key);
                out.push_str("\"\t\t\"");
                out.push_str(v);
                out.push_str("\"\n");
            }
            Value::Block(children) => {
                out.push('"');
                out.push_str(&pair.key);
                out.push_str("\"\n");
                indent(out, depth);
                out.push_str("{\n");
                write_into(out, children, depth + 1);
                indent(out, depth);
                out.push_str("}\n");
            }
        }
    }
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push('\t');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_nested_blocks() {
        let src = "\"a\"\n{\n\t\"k\"\t\t\"v\"\n\t\"b\"\n\t{\n\t\t\"x\"\t\t\"1\"\n\t}\n}\n";
        let pairs = parse(src).unwrap();
        assert_eq!(write(&pairs), src);
    }

    #[test]
    fn accepts_comments_unquoted_and_escapes() {
        let pairs = parse("// hi\nroot { key \"say \\\"hi\\\"\" }").unwrap();
        assert_eq!(
            pairs,
            vec![Pair::block(
                "root",
                vec![Pair::str("key", "say \\\"hi\\\"")]
            )]
        );
    }

    #[test]
    fn reports_line_of_error() {
        let err = parse("\"a\"\n{\n\"k\"\n}").unwrap_err();
        assert_eq!(err.line, 4);
        assert!(parse("\"a\"\n{\n").is_err());
        assert!(parse("}").is_err());
    }
}
