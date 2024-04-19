# Contributors Guide

Welcome to our project! We value your contributions and want to make it easy for you to get involved. This guide outlines our project's workflow, versioning strategy, and commit message conventions.

## Workflow: GitHub Flow

To get started with contributing, please follow the GitHub Flow:

1. **Fork** this repository to your own GitHub account.
2. **Clone** the forked repository to your local machine:
    ```bash
    git clone <your-fork-url>
    git checkout -b your-new-branch-name
    ```
3. **Create a branch** for your modifications:
    ```bash
    git checkout -b feature-branch-name
    ```
4. **Make your changes** and commit them using the [Conventional Commits](#commit-messages) format.
5. **Push** your branch to your fork:
    ```bash
    git push origin feature-branch-name
    ```
6. **Create a pull request** against the original repository’s main branch.
7. **Review** the changes with maintainers and **address any feedback**.

The project maintainers will merge the changes once they are approved. If you have any questions or need help, please feel free to reach out to us.

## Semantic Versioning

Our project adheres to Semantic Versioning (SemVer) principles, which helps us manage releases and dependencies systematically. The version number format is:

MAJOR.MINOR.PATCH

- **MAJOR**: Incompatible API changes,
- **MINOR**: Additions that are backward compatible,
- **PATCH**: Backward-compatible bug fixes.

## Commit Messages

We use the Conventional Commits format for our commit messages to facilitate consistency and changelog generation. Please structure your commits as follows:

<type>(<scope>): <description>

[optional body]

[optional footer]

markdown


### Commit Types <type>

- `feat`: Introduces a new feature.
- `fix`: Patches a bug.
- `docs`: Documentation only changes.
- `style`: Code style update (formatting, semi-colons, etc.).
- `refactor`: Neither fixes a bug nor introduces a feature.
- `perf`: Performance improvements.
- `test`: Adding missing tests or correcting existing ones.
- `chore`: Other changes that don't modify src or test files.

**Example Commit Message**:

```
feat(authentication): add biometric login support

- Support fingerprint and face recognition login.
- Ensure compatibility with mobile devices.
```

Thank you for contributing to our project! We appreciate your effort to follow these guidelines.
