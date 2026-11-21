package relay

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"llm-one-api/internal/convert/ir"
)

const defaultTimeout = 10 * time.Minute

// Upstream is one provider the relay can call.
type Upstream struct {
	Name    string
	Format  ir.Format
	BaseURL string
	APIKey  string
	Headers map[string]string
}

// Route sends models matching Model (a glob, * is the only metacharacter) to an upstream.
// UpstreamModel, when set, is the model name sent upstream; the client still sees its own.
type Route struct {
	Model         string
	Upstream      string
	UpstreamModel string
}

// Config is the relay configuration.
type Config struct {
	Listen    string
	Loss      ir.LossPolicy
	Timeout   time.Duration
	Upstreams []Upstream
	Routes    []Route
}

type fileConfig struct {
	Listen    string `yaml:"listen"`
	Loss      string `yaml:"loss"`
	Timeout   string `yaml:"timeout"`
	Upstreams []struct {
		Name    string            `yaml:"name"`
		Format  string            `yaml:"format"`
		BaseURL string            `yaml:"base_url"`
		APIKey  string            `yaml:"api_key"`
		Headers map[string]string `yaml:"headers"`
	} `yaml:"upstreams"`
	Routes []struct {
		Model         string `yaml:"model"`
		Upstream      string `yaml:"upstream"`
		UpstreamModel string `yaml:"upstream_model"`
	} `yaml:"routes"`
}

var envVar = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`)

func expandEnv(s string) string {
	return envVar.ReplaceAllStringFunc(s, func(m string) string {
		name := envVar.FindStringSubmatch(m)[1]
		if v, ok := os.LookupEnv(name); ok {
			return v
		}
		return m
	})
}

// LoadConfig reads a YAML config file.
func LoadConfig(path string) (Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return Config{}, err
	}
	defer f.Close()
	return ParseConfig(f)
}

// ParseConfig parses and validates YAML config from r.
func ParseConfig(r io.Reader) (Config, error) {
	var raw fileConfig
	dec := yaml.NewDecoder(r)
	dec.KnownFields(true)
	if err := dec.Decode(&raw); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}
	cfg := Config{
		Listen:  expandEnv(raw.Listen),
		Loss:    ir.LossPolicy(expandEnv(raw.Loss)),
		Timeout: defaultTimeout,
	}
	if cfg.Listen == "" {
		cfg.Listen = ":8080"
	}
	if cfg.Loss == "" {
		cfg.Loss = ir.LossAllow
	}
	switch cfg.Loss {
	case ir.LossAllow, ir.LossSafe, ir.LossStrict:
	default:
		return Config{}, fmt.Errorf("loss must be allow, safe, or strict")
	}
	if raw.Timeout != "" {
		d, err := time.ParseDuration(expandEnv(raw.Timeout))
		if err != nil {
			return Config{}, fmt.Errorf("timeout: %w", err)
		}
		if d <= 0 {
			return Config{}, fmt.Errorf("timeout must be positive")
		}
		cfg.Timeout = d
	}
	if len(raw.Upstreams) == 0 {
		return Config{}, fmt.Errorf("at least one upstream is required")
	}
	names := map[string]struct{}{}
	for _, u := range raw.Upstreams {
		name := expandEnv(u.Name)
		format := ir.Format(expandEnv(u.Format))
		base := strings.TrimRight(expandEnv(u.BaseURL), "/")
		if name == "" || base == "" {
			return Config{}, fmt.Errorf("upstream %q needs a name and base_url", name)
		}
		if !format.Valid() {
			return Config{}, fmt.Errorf("upstream %q has unknown format %q", name, u.Format)
		}
		if _, dup := names[name]; dup {
			return Config{}, fmt.Errorf("duplicate upstream %q", name)
		}
		names[name] = struct{}{}
		headers := make(map[string]string, len(u.Headers))
		for k, v := range u.Headers {
			headers[k] = expandEnv(v)
		}
		cfg.Upstreams = append(cfg.Upstreams, Upstream{
			Name:    name,
			Format:  format,
			BaseURL: base,
			APIKey:  expandEnv(u.APIKey),
			Headers: headers,
		})
	}
	for _, rt := range raw.Routes {
		name := expandEnv(rt.Upstream)
		if _, ok := names[name]; !ok {
			return Config{}, fmt.Errorf("route %q refers to unknown upstream %q", rt.Model, name)
		}
		pattern := expandEnv(rt.Model)
		if pattern == "" {
			pattern = "*"
		}
		cfg.Routes = append(cfg.Routes, Route{
			Model:         pattern,
			Upstream:      name,
			UpstreamModel: expandEnv(rt.UpstreamModel),
		})
	}
	if len(cfg.Routes) == 0 {
		if len(cfg.Upstreams) != 1 {
			return Config{}, fmt.Errorf("routes are required when more than one upstream is configured")
		}
		cfg.Routes = []Route{{Model: "*", Upstream: cfg.Upstreams[0].Name}}
	}
	return cfg, nil
}

// match returns the first route whose glob matches model.
func (c Config) match(model string) (Route, Upstream, bool) {
	byName := make(map[string]Upstream, len(c.Upstreams))
	for _, u := range c.Upstreams {
		byName[u.Name] = u
	}
	for _, rt := range c.Routes {
		if matchModel(rt.Model, model) {
			u, ok := byName[rt.Upstream]
			return rt, u, ok
		}
	}
	return Route{}, Upstream{}, false
}

// matchModel reports whether model matches pattern. * is the only metacharacter.
func matchModel(pattern, model string) bool {
	if pattern == "" || pattern == "*" {
		return true
	}
	parts := strings.Split(pattern, "*")
	if len(parts) == 1 {
		return pattern == model
	}
	if !strings.HasPrefix(model, parts[0]) {
		return false
	}
	rest := model[len(parts[0]):]
	for _, p := range parts[1 : len(parts)-1] {
		i := strings.Index(rest, p)
		if i < 0 {
			return false
		}
		rest = rest[i+len(p):]
	}
	return strings.HasSuffix(rest, parts[len(parts)-1])
}
