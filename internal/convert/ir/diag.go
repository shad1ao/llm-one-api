package ir

import (
	"fmt"
	"strings"
)

type Severity string

const (
	SevInfo  Severity = "info"
	SevWarn  Severity = "warn"
	SevError Severity = "error"
)

// Diag records something that could not be represented faithfully.
type Diag struct {
	Severity Severity
	Code     string
	Message  string
}

type Diags []Diag

func (d *Diags) Add(sev Severity, code, msg string) {
	*d = append(*d, Diag{Severity: sev, Code: code, Message: msg})
}

// Worst returns the highest severity present ("" if empty).
func (d Diags) Worst() Severity {
	w := Severity("")
	for _, x := range d {
		switch {
		case x.Severity == SevError:
			return SevError
		case x.Severity == SevWarn:
			w = SevWarn
		case w == "":
			w = SevInfo
		}
	}
	return w
}

func (d Diags) String() string {
	var b strings.Builder
	for i, x := range d {
		if i > 0 {
			b.WriteString("; ")
		}
		fmt.Fprintf(&b, "%s:%s %s", x.Severity, x.Code, x.Message)
	}
	return b.String()
}

// LossPolicy decides whether a lossy request conversion is acceptable.
type LossPolicy string

const (
	LossAllow  LossPolicy = "allow"  // never fail
	LossSafe   LossPolicy = "safe"   // fail on error diagnostics
	LossStrict LossPolicy = "strict" // fail on warn or error diagnostics
)

// Check returns an error if the diagnostics violate the policy.
func (p LossPolicy) Check(d Diags) error {
	w := d.Worst()
	switch p {
	case LossSafe:
		if w == SevError {
			return fmt.Errorf("lossy conversion rejected: %s", d)
		}
	case LossStrict:
		if w == SevError || w == SevWarn {
			return fmt.Errorf("lossy conversion rejected: %s", d)
		}
	}
	return nil
}
