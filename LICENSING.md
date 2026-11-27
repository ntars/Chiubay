# 许可分层说明（LICENSING）

本项目采用**按目录分层**的许可，而不是「二选一双许可」：代码用 Apache-2.0，文档与设计稿用 MIT。

## 一、分层映射

| 范围 | 许可证 | 理由 |
| --- | --- | --- |
| `src/`、`experiments/`、`tools/` 等代码目录（含未来新增代码） | Apache-2.0 | 含专利授权（§3）与商标保留（§6） |
| `README.md`、`docs/`、`LICENSING.md`、设计稿与实验记录 | MIT | 文档要能被自由引用、翻译与再发布 |

当前仓库尚无正式源码时，Apache-2.0 覆盖未来提交的代码；已存在的文档与设计内容按 MIT 授权。

## 二、我要用你的代码，该看哪份

| 我要做什么 | 看哪份 |
| --- | --- |
| 复制、修改、分发源码 | [LICENSE](./LICENSE)（Apache-2.0） |
| 引用文档、设计或实验结论 | [LICENSE-MIT](./LICENSE-MIT)（MIT） |
| 使用项目名称或图标 | 不在许可范围内，需另行授权 |

## 三、硬规则

1. 新增源文件头须带 SPDX 标识，且与所在目录映射一致。
2. 依赖白名单只放 MIT / Apache-2.0 / BSD / ISC / Unicode-3.0；不引入 GPL / AGPL / SSPL。
3. 项目名称与图标不随代码授权。
4. 贡献采用 DCO（提交带 `Signed-off-by:`）。

Copyright © 2026 星玖工作室 Ntarstudio.
