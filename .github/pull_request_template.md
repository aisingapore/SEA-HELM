<!-- markdownlint-disable-file MD041 -->

## Description

> Should include a concise description of the changes (bug or feature), it's impact, along with a summary of the solution
> If it is a bug, please provide details on the environment the bug is found, and detailed steps to recreate the bug.

## PR Checklist

> Use the check-list below to ensure your branch is ready for PR. If the item is not applicable, leave it blank.

### General check-list

- [ ] I am merging to the dev branch
- [ ] I have updated the documentation accordingly.
- [ ] I have added tests to cover my changes.
- [ ] All new and existing tests passed.
- [ ] My code follows the code style of this project.
- [ ] I ran the lint checks which produced no new errors nor warnings for my changes.

### If a new evaluation is added

- [ ] I have run the evaluation on a few models and the responses for a few models are expected.
- [ ] I have checked through the dataset and did not identify any issues with the data quality.
- [ ] I have checked the LLM judges and they are scoring the outputs correctly for a range of responses.
- [ ] I have tested that the metric works and the scores make sense.

## Does This Introduce a Breaking Change?

> If this introduces a breaking change, please describe the impact and migration path for existing applications below.

- [ ] Yes
- [ ] No

## Testing

> Instructions for testing and validation of your code:
>
> - Which test sets were used.
> - Description of test scenarios that you have tried.

## Any Relevant Logs or Outputs

> Use this section to attach pictures that demonstrates your changes working / healthy
>
> - If you are printing something, show a screenshot

## Other Information or Known Dependencies

> Any other information or known dependencies that is important to this PR.
>
> - TODO that are to be done after this PR.
