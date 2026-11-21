// Package all imports every wire codec so each one registers itself.
// Programs that convert or relay traffic should import this package for the side effect.
package all

import (
	_ "llm-one-api/internal/convert/claude"
	_ "llm-one-api/internal/convert/gemini"
	_ "llm-one-api/internal/convert/openaichat"
	_ "llm-one-api/internal/convert/openairesponses"
)
